"""Fused fp16 attention (FlashAttention-2 style) for the tinygrad OWLv2 port, as a hand-written CUDA kernel.

tinygrad's ``scaled_dot_product_attention`` materialises the full [B, H, L, L] score
matrix and makes three more passes over it for the softmax; for the vision tower
(L = 3616 for base) that is ~600 MB per layer, which made attention 2/3 of the fp16
forward. This kernel keeps the scores in registers with an online softmax (fp32
accumulation, fp16 tensor-core mma.sync), so nothing L x L ever reaches memory.

It is injected through ``Tensor.custom_kernel`` as a pre-rendered PROGRAM, which
tinygrad compiles with its own nvrtc pipeline and runs inside TinyJit. Only the
CUDA C renderer (NV / CUDA devices, sm_80+) and head_dim 64 are supported.
"""
import math

from tinygrad import Device, Tensor, dtypes
from tinygrad.renderer import Estimates
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp

# q, k, v and out are [B, L, H*64] fp16 (the projection layout, no head transposes).
# Grid (ceil(L/(64*MT)), H, B), 4 warps of 16*MT query rows each; K/V tiles of 64 keys are
# double-buffered in shared memory with cp.async. Keys at positions >= NVALID are masked.
_SRC = r"""
#include <cuda_fp16.h>
#define INFINITY (__int_as_float(0x7f800000))
#define L_ {L}
#define NVALID {NVALID}
#define DM {DM}
#define LDS 72
#define TILE (64 * LDS)
#define MT {MT}
__device__ __forceinline__ void mma16816(float* c, const unsigned* a, const unsigned* b) {{
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}}
__device__ __forceinline__ void ldsm_x4(unsigned* r, const half* p) {{
  unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%0,%1,%2,%3}}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}}
__device__ __forceinline__ void ldsm_x4_t(unsigned* r, const half* p) {{
  unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%0,%1,%2,%3}}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}}
__device__ __forceinline__ void cp16(half* dst, const half* src, bool pred) {{
  unsigned a = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(a), "l"(src), "r"(pred ? 16 : 0));
}}
__device__ __forceinline__ float ex2(float x) {{ float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }}
__device__ __forceinline__ unsigned pack2(float x, float y) {{ __half2 h = __floats2half2_rn(x, y); return *(unsigned*)&h; }}

extern "C" __global__ void __launch_bounds__(128) {NAME}(half* out, const half* q, const half* k, const half* v) {{
  __shared__ __align__(16) half sm[4 * TILE];  // [stage][K|V][key][dim]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const size_t base = (size_t)blockIdx.z * L_ * DM + blockIdx.y * 64;
  // This thread's two query rows in each of the warp's MT 16-row tiles.
  int rows[MT][2];
  #pragma unroll
  for (int mt = 0; mt < MT; mt++) {{
    rows[mt][0] = blockIdx.x * 64 * MT + (warp * MT + mt) * 16 + g;
    rows[mt][1] = rows[mt][0] + 8;
  }}

  auto load_tile = [&](int kb, int stage) {{
    half* ks = sm + stage * 2 * TILE; half* vs = ks + TILE;
    #pragma unroll
    for (int i = tid; i < 64 * 8; i += 128) {{
      const int row = i >> 3, col = (i & 7) * 8, key = kb + row;
      const size_t off = base + (size_t)(key < L_ ? key : 0) * DM + col;
      cp16(ks + row * LDS + col, k + off, key < L_);
      cp16(vs + row * LDS + col, v + off, key < L_);
    }}
    asm volatile("cp.async.commit_group;");
  }};
  load_tile(0, 0);

  unsigned qa[MT][4][4];
  #pragma unroll
  for (int mt = 0; mt < MT; mt++) {{
    const int r0 = rows[mt][0], r1 = rows[mt][1];
    #pragma unroll
    for (int kk = 0; kk < 4; kk++) {{
      const int c = kk * 16 + t * 2;
      qa[mt][kk][0] = r0 < L_ ? *(const unsigned*)(q + base + (size_t)r0 * DM + c) : 0u;
      qa[mt][kk][1] = r1 < L_ ? *(const unsigned*)(q + base + (size_t)r1 * DM + c) : 0u;
      qa[mt][kk][2] = r0 < L_ ? *(const unsigned*)(q + base + (size_t)r0 * DM + c + 8) : 0u;
      qa[mt][kk][3] = r1 < L_ ? *(const unsigned*)(q + base + (size_t)r1 * DM + c + 8) : 0u;
    }}
  }}
  float o[MT][8][4], m[MT][2], l[MT][2];
  #pragma unroll
  for (int mt = 0; mt < MT; mt++) {{
    m[mt][0] = m[mt][1] = -INFINITY; l[mt][0] = l[mt][1] = 0.f;
    #pragma unroll
    for (int n = 0; n < 8; n++) o[mt][n][0] = o[mt][n][1] = o[mt][n][2] = o[mt][n][3] = 0.f;
  }}
  const float sl2 = {SCALE}f * 1.4426950408889634f;
  const int mi = lane >> 3, mr = lane & 7;
  constexpr int NT = (NVALID + 63) / 64;

  for (int it = 0; it < NT; it++) {{
    const int kb = it * 64;
    if (it + 1 < NT) {{ load_tile(kb + 64, (it + 1) & 1); asm volatile("cp.async.wait_group 1;"); }}
    else asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    const half* ks = sm + (it & 1) * 2 * TILE; const half* vs = ks + TILE;

    // S = Q K^T: each K fragment from shared memory feeds all MT query tiles.
    float s[MT][8][4];
    #pragma unroll
    for (int n = 0; n < 8; n++) {{
      #pragma unroll
      for (int mt = 0; mt < MT; mt++) s[mt][n][0] = s[mt][n][1] = s[mt][n][2] = s[mt][n][3] = 0.f;
      #pragma unroll
      for (int kk = 0; kk < 4; kk += 2) {{
        unsigned bb[4];
        ldsm_x4(bb, ks + (n * 8 + mr) * LDS + (kk + (mi >> 1)) * 16 + (mi & 1) * 8);
        #pragma unroll
        for (int mt = 0; mt < MT; mt++) {{ mma16816(s[mt][n], qa[mt][kk], bb); mma16816(s[mt][n], qa[mt][kk + 1], bb + 2); }}
      }}
    }}

    // Online softmax in fp32 (base 2), per query tile.
    #pragma unroll
    for (int mt = 0; mt < MT; mt++) {{
      float mx0 = m[mt][0], mx1 = m[mt][1];
      #pragma unroll
      for (int n = 0; n < 8; n++) {{
        #pragma unroll
        for (int j = 0; j < 2; j++) {{
          const bool valid = kb + n * 8 + t * 2 + j < NVALID;
          s[mt][n][j] = valid ? s[mt][n][j] * sl2 : -INFINITY;
          s[mt][n][2 + j] = valid ? s[mt][n][2 + j] * sl2 : -INFINITY;
          mx0 = fmaxf(mx0, s[mt][n][j]); mx1 = fmaxf(mx1, s[mt][n][2 + j]);
        }}
      }}
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
      const float a0 = ex2(m[mt][0] - mx0), a1 = ex2(m[mt][1] - mx1);
      m[mt][0] = mx0; m[mt][1] = mx1;
      float rs0 = 0.f, rs1 = 0.f;
      #pragma unroll
      for (int n = 0; n < 8; n++) {{
        #pragma unroll
        for (int j = 0; j < 2; j++) {{
          s[mt][n][j] = ex2(s[mt][n][j] - mx0); rs0 += s[mt][n][j];
          s[mt][n][2 + j] = ex2(s[mt][n][2 + j] - mx1); rs1 += s[mt][n][2 + j];
        }}
        o[mt][n][0] *= a0; o[mt][n][1] *= a0; o[mt][n][2] *= a1; o[mt][n][3] *= a1;
      }}
      l[mt][0] = l[mt][0] * a0 + rs0; l[mt][1] = l[mt][1] * a1 + rs1;
    }}

    // O += P V: P's accumulator layout is reused as the A operand; each V fragment feeds all MT tiles.
    #pragma unroll
    for (int j = 0; j < 4; j++) {{
      unsigned pa[MT][4];
      #pragma unroll
      for (int mt = 0; mt < MT; mt++) {{
        pa[mt][0] = pack2(s[mt][2*j][0], s[mt][2*j][1]);     pa[mt][1] = pack2(s[mt][2*j][2], s[mt][2*j][3]);
        pa[mt][2] = pack2(s[mt][2*j+1][0], s[mt][2*j+1][1]); pa[mt][3] = pack2(s[mt][2*j+1][2], s[mt][2*j+1][3]);
      }}
      #pragma unroll
      for (int n = 0; n < 8; n += 2) {{
        unsigned bb[4];
        ldsm_x4_t(bb, vs + (j * 16 + (mi & 1) * 8 + mr) * LDS + (n + (mi >> 1)) * 8);
        #pragma unroll
        for (int mt = 0; mt < MT; mt++) {{ mma16816(o[mt][n], pa[mt], bb); mma16816(o[mt][n + 1], pa[mt], bb + 2); }}
      }}
    }}
    __syncthreads();
  }}

  #pragma unroll
  for (int mt = 0; mt < MT; mt++) {{
    float l0 = l[mt][0], l1 = l[mt][1];
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = 1.f / l0, i1 = 1.f / l1;
    const int r0 = rows[mt][0], r1 = rows[mt][1];
    #pragma unroll
    for (int n = 0; n < 8; n++) {{
      const int c = n * 8 + t * 2;
      if (r0 < L_) *(unsigned*)(out + base + (size_t)r0 * DM + c) = pack2(o[mt][n][0] * i0, o[mt][n][1] * i0);
      if (r1 < L_) *(unsigned*)(out + base + (size_t)r1 * DM + c) = pack2(o[mt][n][2] * i1, o[mt][n][3] * i1);
    }}
  }}
}}
"""


def flash_attention_supported(x: Tensor, head_dim: int) -> bool:
    """True when :func:`flash_attention` can run on ``x``'s device and dtype."""
    if x.dtype != dtypes.half or head_dim != 64 or not isinstance(x.device, str):
        return False
    renderer = Device[x.device].renderer
    if not isinstance(renderer, CUDARenderer):
        return False
    return int(renderer.target.arch[3:]) >= 80


def flash_attention(q: Tensor, k: Tensor, v: Tensor, num_heads: int, num_valid: int | None = None,
                    m_tiles: int = 2) -> Tensor:
    """Softmax attention over ``[B, L, num_heads * 64]`` fp16 q/k/v, returning the same layout.

    Keys at positions ``>= num_valid`` are masked out (the vision tower's sequence padding).
    Each warp handles ``m_tiles`` 16-row query tiles, so a block covers ``64 * m_tiles`` queries.
    """
    B, L, DM = q.shape
    if DM != num_heads * 64 or q.dtype != dtypes.half:
        raise ValueError(f"expected fp16 [B, L, {num_heads} * 64], got {q.dtype} {q.shape}")
    num_valid = L if num_valid is None else num_valid
    name = f"flash_attn_{B}_{L}_{num_valid}_{num_heads}_mt{m_tiles}"
    src = _SRC.format(NAME=name, L=L, NVALID=num_valid, DM=DM, SCALE=repr(1 / math.sqrt(64)), MT=m_tiles)
    target = Device[q.device].renderer.target

    def fxn(*_params):
        info = ProgramInfo(name=name, global_size=(math.ceil(L / (64 * m_tiles)), num_heads, B), local_size=(128, 1, 1),
                           globals=(0, 1, 2, 3), outs=(0,), ins=(1, 2, 3), target=target)
        # QK^T and PV matmuls; q/k/v read and out written once from the kernel's point of view.
        est = Estimates(ops=4 * B * num_heads * L * num_valid * 64, lds=4 * B * L * DM * 2, mem=4 * B * L * DM * 2)
        sink = UOp.sink(arg=KernelInfo(name=name, estimates=est))
        return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=()), UOp(Ops.SOURCE, arg=src)), arg=info)

    out = Tensor.empty(B, L, DM, dtype=dtypes.half, device=q.device)
    return out.custom_kernel(q.contiguous(), k.contiguous(), v.contiguous(), fxn=fxn)[0]
