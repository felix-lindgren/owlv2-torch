"""Flash-attention backward with fp16 accumulation in its long reductions.

GeForce Blackwell runs fp32-accumulate fp16 MMA at half rate, and FA2's backward
is the largest single kernel cost of an OWLv2 training step (~112 ms of ~490 on
the 5070 Ti). This backward accumulates dV, dK and dQ in fp16 within a key/query
tile and in fp32 across tiles; the score recompute S = QK^T and dP = dO V^T stay
fp32, because real OWLv2 logits reach |s| ~ 200 and fp16 there is 10-20x FA2's
error on dK/dV. On the base shape (B=8, N=3601, H=12, D=64) it is 1.31x FA2's
backward at 1.2-1.4x FA2's error against fp32.

The forward is PyTorch's own FA2 kernel, unchanged: its natural-log logsumexp
feeds this backward after a change to base 2. Outputs, eval and inference are
therefore bit-identical; only gradients differ.

:func:`install` routes eligible ``torch.nn.functional.scaled_dot_product_attention``
calls (fp16, head dim 64, no mask or dropout, gradients needed) through it.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

LOG2E = 1.0 / math.log(2.0)
# (BLOCK_M, BLOCK_N, num_warps, num_stages), tuned on the 5070 Ti at the base shape.
BWD_CONFIG = (16, 128, 4, 2)


@triton.jit
def _bwd_preprocess(O, DO, DELTA, sob, soh, son, sdb, sdh, sdn, H, N,
                    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr):
    """delta = rowsum(dO * O), the softmax-backward correction term."""
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask = offs_m[:, None] < N
    o = tl.load(O + b * sob + h * soh + offs_m[:, None] * son + offs_d[None, :], mask=mask, other=0.0)
    do = tl.load(DO + b * sdb + h * sdh + offs_m[:, None] * sdn + offs_d[None, :], mask=mask, other=0.0)
    tl.store(DELTA + bh * N + offs_m, tl.sum(o.to(tl.float32) * do.to(tl.float32), 1), mask=offs_m < N)


@triton.jit
def _bwd_kernel(Q, K, V, DO, LSE, DELTA, DK, DV, DQ_ACC, sm_scale,
                sqb, sqh, sqn, skb, skh, skn, svb, svh, svn, sdb, sdh, sdn,
                skgb, skgh, skgn, svgb, svgh, svgn, H, N,
                HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """FA2-style fused backward: one program per key block keeps dK/dV in registers
    and adds its share of dQ to an fp32 buffer with atomics."""
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    nmask = offs_n[:, None] < N
    k = tl.load(K + b * skb + h * skh + offs_n[:, None] * skn + offs_d[None, :], mask=nmask, other=0.0)
    k = (k * sm_scale).to(tl.float16)
    v = tl.load(V + b * svb + h * svh + offs_n[:, None] * svn + offs_d[None, :], mask=nmask, other=0.0)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    qT_ptrs = Q + b * sqb + h * sqh + offs_m[None, :] * sqn + offs_d[:, None]
    do_ptrs = DO + b * sdb + h * sdh + offs_m[:, None] * sdn + offs_d[None, :]
    dq_ptrs = DQ_ACC + bh * N * HEAD_DIM + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
    for start_m in range(0, N, BLOCK_M):
        mmask = (start_m + offs_m) < N
        qT = tl.load(qT_ptrs + start_m * sqn, mask=mmask[None, :], other=0.0)
        do = tl.load(do_ptrs + start_m * sdn, mask=mmask[:, None], other=0.0)
        lse = tl.load(LSE + bh * N + start_m + offs_m, mask=mmask, other=float("inf"))
        delta = tl.load(DELTA + bh * N + start_m + offs_m, mask=mmask, other=0.0)
        # Short (D=64) reductions in fp32; the long ones below in fp16 per tile.
        pT = tl.math.exp2(tl.dot(k, qT) * 1.4426950408889634 - lse[None, :])
        dv += tl.dot(pT.to(tl.float16), do, out_dtype=tl.float16).to(tl.float32)
        dsT = (pT * (tl.dot(v, tl.trans(do)) - delta[None, :])).to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT), out_dtype=tl.float16).to(tl.float32)
        dq = tl.dot(tl.trans(dsT), k, out_dtype=tl.float16).to(tl.float32)
        tl.atomic_add(dq_ptrs + start_m * HEAD_DIM, dq, mask=mmask[:, None], sem="relaxed")
    dk = dk * sm_scale
    tl.store(DK + b * skgb + h * skgh + offs_n[:, None] * skgn + offs_d[None, :], dk.to(tl.float16), mask=nmask)
    tl.store(DV + b * svgb + h * svgh + offs_n[:, None] * svgn + offs_d[None, :], dv.to(tl.float16), mask=nmask)


def _strides(t: torch.Tensor) -> tuple[int, int, int]:
    return t.stride(0), t.stride(1), t.stride(2)


def _unit_last_stride(t: torch.Tensor) -> torch.Tensor:
    return t if t.stride(-1) == 1 else t.contiguous()


def _like_model(t: torch.Tensor) -> torch.Tensor:
    """``(B, H, N, D)`` laid out as the model's ``(B, N, H, D)`` projections."""
    batch, heads, tokens, dim = t.shape
    return t.new_empty((batch, tokens, heads, dim)).transpose(1, 2)


@torch.library.custom_op("owlv2::flash_attention_fwd", mutates_args=())
def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False, scale=scale)[:2]
    return out, lse


@_forward.register_fake
def _(q, k, v, scale):
    batch, heads, tokens, _ = q.shape
    return _like_model(q), q.new_empty((batch, heads, tokens), dtype=torch.float32)


@torch.library.custom_op("owlv2::flash_attention_bwd_fp16acc", mutates_args=())
def _backward_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor, lse: torch.Tensor,
    grad_out: torch.Tensor, scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v, out, grad_out = (_unit_last_stride(t) for t in (q, k, v, out, grad_out))
    batch, heads, tokens, dim = q.shape
    lse2 = (lse[..., :tokens] * LOG2E).contiguous()
    delta = torch.empty((batch, heads, tokens), device=q.device, dtype=torch.float32)
    _bwd_preprocess[(triton.cdiv(tokens, 128), batch * heads)](
        out, grad_out, delta, *_strides(out), *_strides(grad_out), heads, tokens, dim, 128, num_warps=4
    )
    dq_acc = torch.zeros((batch, heads, tokens, dim), device=q.device, dtype=torch.float32)
    dq, dk, dv = _like_model(q), _like_model(k), _like_model(v)
    block_m, block_n, warps, stages = BWD_CONFIG
    _bwd_kernel[(triton.cdiv(tokens, block_n), batch * heads)](
        q, k, v, grad_out, lse2, delta, dk, dv, dq_acc, scale,
        *_strides(q), *_strides(k), *_strides(v), *_strides(grad_out), *_strides(dk), *_strides(dv),
        heads, tokens, dim, block_m, block_n, num_warps=warps, num_stages=stages,
    )
    dq.copy_(dq_acc)
    return dq, dk, dv


@_backward_op.register_fake
def _(q, k, v, out, lse, grad_out, scale):
    return _like_model(q), _like_model(k), _like_model(v)


def _setup_context(ctx, inputs, output):
    q, k, v, scale = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)
    ctx.scale = scale


def _backward(ctx, grad_out, grad_lse):
    q, k, v, out, lse = ctx.saved_tensors
    dq, dk, dv = torch.ops.owlv2.flash_attention_bwd_fp16acc(q, k, v, out, lse, grad_out, ctx.scale)
    return dq, dk, dv, None


_forward.register_autograd(_backward, setup_context=_setup_context)

_original_sdpa = torch.nn.functional.scaled_dot_product_attention


def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
    if (
        attn_mask is None and dropout_p == 0.0 and not is_causal and not kwargs
        and query.dtype == torch.float16 and query.shape[-1] == 64
        and key.shape == query.shape and value.shape == query.shape
        and torch.is_grad_enabled()
        and (query.requires_grad or key.requires_grad or value.requires_grad)
    ):
        return torch.ops.owlv2.flash_attention_fwd(
            query, key, value, query.shape[-1] ** -0.5 if scale is None else scale
        )[0]
    return _original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                          is_causal=is_causal, scale=scale, **kwargs)


def install() -> None:
    """Route eligible SDPA calls through the FA2 forward + fp16-accumulate backward."""
    torch.nn.functional.scaled_dot_product_attention = _sdpa
