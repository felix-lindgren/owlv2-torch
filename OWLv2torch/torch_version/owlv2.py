import functools
import math
import warnings

from huggingface_hub.file_download import hf_hub_download
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from safetensors import safe_open

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from torch.utils.checkpoint import checkpoint
from PIL import Image
from scipy import ndimage as ndi
from scipy import sparse as sps

from OWLv2torch.utils.hf_hub_utils import find_safetensors_in_cache
from OWLv2torch.utils.query_batching import normalize_detection_queries
from OWLv2torch.torch_version.prototypes import VisualPrototypeBank

from typing import Optional, Tuple

OPENAI_CLIP_MEAN: list[float] = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD: list[float] = [0.26862954, 0.26130258, 0.27577711]


def build_causal_padding_mask(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a combined causal + padding additive attention mask on-device.

    Padding positions are always derived from ``input_ids == 0`` (the CLIP
    tokenizer uses pad id 0). This matches the behavior of the previous
    ``process_sequences`` implementation, which silently ignored any
    ``attention_mask`` passed in — some existing callers pass an inverted
    mask (True where padded), so we deliberately don't trust the argument.
    It's accepted for API compatibility only.

    Args:
        input_ids: [B, L] token ids.
        attention_mask: accepted for API compatibility; ignored.
        dtype: dtype of the returned additive mask.

    Returns:
        [B, 1, L, L] additive mask (0 where attended, large negative where masked).
    """
    del attention_mask  # intentionally ignored — see docstring.
    B, L = input_ids.shape
    device = input_ids.device
    mask_value = torch.finfo(dtype).min

    # Causal mask: upper triangle (excluding diagonal) is masked.
    causal = torch.triu(
        torch.full((L, L), mask_value, dtype=dtype, device=device),
        diagonal=1,
    )
    causal = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L)

    # Padding: CLIP tokenizer uses id 0 for pad.
    pad_positions = input_ids == 0  # [B, L], True where padded
    combined = causal.masked_fill(pad_positions.view(B, 1, 1, L), mask_value)
    return combined
class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input):
        return input * torch.sigmoid(1.702 * input)

class SquarePad:
    def __call__(self, image):
        h, w = image.shape[-2:]
        max_wh = np.max([w, h])
        hp = int(max_wh - w)
        vp = int(max_wh - h)
        padding = (0, 0, hp, vp)
        return TF.pad(image, padding, 0.5, 'constant')


class ScipyResize:
    """Resize using scipy to match HuggingFace OWLv2 preprocessing exactly."""
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        # image is a torch tensor (C, H, W) in float
        arr = image.permute(1, 2, 0).numpy()  # (H, W, C)
        input_shape = arr.shape
        output_shape = (self.size, self.size, arr.shape[2])
        factors = np.divide(input_shape, output_shape)
        # Gaussian anti-aliasing (matches HF default)
        anti_aliasing_sigma = np.maximum(0, (factors - 1) / 2)
        filtered = ndi.gaussian_filter(arr, anti_aliasing_sigma, cval=0, mode="mirror")
        zoom_factors = [1 / f for f in factors]
        out = ndi.zoom(filtered, zoom_factors, order=1, mode="mirror", cval=0, grid_mode=True)
        out = np.clip(out, arr.min(), arr.max())
        return torch.from_numpy(out).permute(2, 0, 1)  # back to (C, H, W)


def _mirror_index(idx, n):
    """Fold indices into ``[0, n)`` with scipy.ndimage's 'mirror' boundary (d c b | a b c d | c b a)."""
    if n == 1:
        return np.zeros_like(idx)
    period = 2 * n - 2
    idx = np.abs(idx) % period
    return np.where(idx >= n, period - idx, idx)


def _scipy_resize_axis(content, padded, out, scale):
    """One axis of ScipyResize as an ``(out, content)`` sparse CSR operator.

    The axis is ``padded`` samples long after SquarePad, of which the first
    ``content`` are image and the rest the 0.5 fill. Each output sample is
    ndi.zoom's order-1 interpolation (grid_mode, mirror) of ndi.gaussian_filter's
    output (truncate=4, mirror), i.e. a short list of taps into the padded axis.
    Taps landing on the fill are dropped from the operator; the returned
    per-row weight on the image says how much of each row they took.
    """
    rows, cols, weights, image_weight = _scipy_resize_taps(content, padded, out)
    return _to_torch_csr(rows, cols, weights * scale, (out, content)), torch.from_numpy(image_weight).float()


def _scipy_resize_taps(content, padded, out):
    """``_scipy_resize_axis``'s operator as ``(rows, cols, weights)`` taps, plus the per-row image weight."""
    factor = padded / out
    sigma = max(0.0, (factor - 1) / 2)
    src = (np.arange(out) + 0.5) * factor - 0.5
    lo = np.floor(src)
    frac = src - lo
    cols = _mirror_index(np.stack([lo, lo + 1], 1).astype(np.int64), padded)
    weights = np.stack([1 - frac, frac], 1)
    if sigma > 1e-15:  # gaussian_filter skips axes with a zero sigma
        radius = int(4.0 * sigma + 0.5)
        offsets = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 / sigma**2 * offsets**2)
        kernel /= kernel.sum()
        cols = _mirror_index(cols[:, :, None] + offsets, padded)
        weights = weights[:, :, None] * kernel
    cols = cols.reshape(out, -1)
    weights = weights.reshape(out, -1)
    rows = np.broadcast_to(np.arange(out)[:, None], cols.shape)
    keep = cols < content
    rows, cols, weights = rows[keep], cols[keep], weights[keep]
    return rows, cols, weights, np.bincount(rows, weights, minlength=out)


def _to_torch_csr(rows, cols, weights, shape):
    """float32 sparse CSR tensor from COO taps; repeated taps are summed."""
    # torch flags sparse tensors as beta and warns that invariant checks are off.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse")
        return torch.sparse_coo_tensor(
            torch.from_numpy(np.stack([rows, cols]).astype(np.int64)), torch.from_numpy(weights), shape
        ).coalesce().to(torch.float32).to_sparse_csr()


@functools.lru_cache(maxsize=32)
def _scipy_resize_operators(h, w, size, scale, device):
    n = max(h, w)
    ay, ay_weight = _scipy_resize_axis(h, n, size, scale)
    ax, ax_weight = _scipy_resize_axis(w, n, size, 1.0)
    # 0.5 fill of the padded margin, laid out like the (S_x, S_y) result below.
    fill = 0.5 * (1 - torch.outer(ax_weight, ay_weight)).unsqueeze(-1)
    return ay.to(device), ax.to(device), fill.to(device)


@functools.lru_cache(maxsize=32)
def _preprocess_operators(h, w, side, rows, cols, blur, scale, device):
    """``preprocess_image``'s fast path for an ``h`` x ``w`` image as cached operators.

    SquarePad + ScipyResize to ``side``, the optional Gaussian ``blur``
    (TF.gaussian_blur's kernel and reflect border, over the whole ``side``
    square) and the crop to ``rows`` x ``cols`` are all linear, so each axis
    composes into one sparse operator plus a fill term for the padding, and the
    normalisation becomes a per-channel scale and shift. Only the kept rectangle
    is computed and the blur takes no pass of its own: a 640x480 PIL image takes
    0.40 instead of 0.73 ms to CUDA at blur 0.5 and long side 768 (5070 Ti),
    within ~1.3e-6 of the separate steps. ScipyResize's clip to the input range
    is left out; every tap weight is non-negative, so it only ever trimmed
    rounding (and cost two device syncs).
    """
    n = max(h, w)
    mix = sps.identity(side, format="csr")
    if blur > 0:
        radius = max(1, int(3 * blur + 0.5))
        offsets = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (offsets / blur) ** 2)
        taps = _mirror_index(np.arange(side)[:, None] + offsets, side)
        mix = sps.csr_matrix(
            (np.broadcast_to(kernel / kernel.sum(), taps.shape).ravel(),
             (np.repeat(np.arange(side), len(offsets)), taps.ravel())),
            shape=(side, side),
        )

    def axis(content, out, axis_scale):
        r, c, weights, image_weight = _scipy_resize_taps(content, n, side)
        resize = sps.csr_matrix((weights * axis_scale, (r, c)), shape=(side, content))
        op = (mix[:out] @ resize).tocoo()
        return _to_torch_csr(op.row, op.col, op.data, op.shape).to(device), mix[:out] @ image_weight

    ay, wy = axis(h, rows, scale)
    ax, wx = axis(w, cols, 1.0)
    # The 0.5 padding after resize and blur, laid out like the (cols, rows) result.
    fill = torch.from_numpy(0.5 * (1 - np.outer(wx, wy))).float().unsqueeze(-1).to(device)
    mean, std = torch.tensor(OPENAI_CLIP_MEAN), torch.tensor(OPENAI_CLIP_STD)
    return ay, ax, fill, (1 / std).to(device), (-mean / std).to(device)


class TorchScipyResize:
    """SquarePad + ScipyResize in one step, as two sparse matmuls in torch.

    Matches ``ScipyResize(SquarePad(image))`` to float32 rounding (~1e-4 of a
    grey level) at a fraction of the cost, and faster than torchvision's
    antialiased resize at the large end since the padded square is never
    materialised. The operators depend only on the image size and are cached.

    Accepts a PIL image, or a ``(C, H, W)`` tensor: integer images are scaled to
    ``[0, 1]`` as ``ToDtype(scale=True)`` does, float images are taken as already
    in range. Returns a float32 ``(C, size, size)`` tensor.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        if isinstance(image, Image.Image):
            hwc = torch.from_numpy(np.array(image))
            if hwc.ndim == 2:
                hwc = hwc.unsqueeze(-1)
        else:
            hwc = image.as_subclass(torch.Tensor).permute(1, 2, 0)
        h, w, c = hwc.shape
        s = self.size
        scale = 1.0 if hwc.is_floating_point() else 1.0 / torch.iinfo(hwc.dtype).max
        ay, ax, fill = _scipy_resize_operators(h, w, s, scale, hwc.device)
        # Contract rows on the (H, W*C) view, then columns on the (W, S*C) view.
        y = torch.sparse.mm(ay, hwc.reshape(h, w * c).float())
        y = y.view(s, w, c).transpose(0, 1).reshape(w, s * c)
        out = torch.sparse.mm(ax, y).view(s, s, c) + fill
        # ScipyResize clips to the range of the padded input.
        lo, hi = (v.item() * scale for v in torch.aminmax(hwc))
        if h != w:
            lo, hi = min(lo, 0.5), max(hi, 0.5)
        return out.clamp_(lo, hi).permute(2, 1, 0).contiguous()


class Attention(nn.Module): 
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # SDPA accepts this standard strided head layout directly. Materialising
        # contiguous Q/K/V tensors here adds three full-sequence copies per layer.
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)


    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_attention_mask: Optional[torch.Tensor] = None,
        extra_states: Optional[torch.Tensor] = None,
        window=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, _ = hidden_states.size()
        # project the hidden states to the query, key, and value states
        query_states = self.q_proj(hidden_states) #* self.scale
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        if extra_states is not None:
            key_states = torch.cat([key_states, self.k_proj(extra_states)], dim=1)
            value_states = torch.cat([value_states, self.v_proj(extra_states)], dim=1)
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)
        query_states = self._shape(query_states, tgt_len, bsz)
        if window is not None:
            grid, size, registers = window
            out = window_attention(query_states, key_states, value_states, grid, size, self.scale, registers)
            out = out.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
            return self.out_proj(out), None
        
        # compute the attention weights
        sdpa_attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            #is_causal=(causal_attention_mask is not None),
            scale=self.scale
        )

        sdpa_attn_output = sdpa_attn_output.transpose(1, 2)
        sdpa_attn_output = sdpa_attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(sdpa_attn_output)
        
        return attn_output, None

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_dim, dropout):
        super().__init__()

        self.self_attn = Attention(hidden_size, num_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            QuickGELUActivation(),
            nn.Linear(mlp_dim, hidden_size),
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-5)


    def forward(self, x, mask=None, extra=None, window=None):
        residual = x
        if window is not None:
            grid, size, num_registers = window
            registers = None
            if num_registers:
                # The highest-norm patch tokens act as registers: they keep global attention.
                with torch.no_grad():
                    registers = x[:, 1:].norm(dim=-1).topk(num_registers, dim=1).indices + 1
            window = (grid, size, registers)
        
        x = self.layer_norm1(x)
        x, _ = self.self_attn(x, mask, None if extra is None else self.layer_norm1(extra), window)
        x = x + residual
        
        residual = x
        
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + residual

        return x

def window_attention(q, k, v, grid, size, scale, registers=None):
    """Patch queries attend within their ``size`` x ``size`` window, plus the CLS key and
    the ``registers`` tokens (sequence indices, [B, R]); CLS and registers attend globally."""
    batch, heads, _, dim = q.shape
    count = grid // size

    def windows(t):
        return (
            t.reshape(batch, heads, count, size, count, size, dim)
            .permute(0, 2, 4, 1, 3, 5, 6)
            .reshape(batch * count * count, heads, size * size, dim)
        )

    def per_window(t):  # [B, H, R, D] -> [B * windows, H, R, D]
        return t.unsqueeze(1).expand(-1, count * count, -1, -1, -1).reshape(batch * count * count, heads, -1, dim)

    def take(t, index):
        return t.gather(2, index[:, None, :, None].expand(-1, heads, -1, dim))

    shared = torch.zeros(batch, 1, dtype=torch.long, device=q.device)
    if registers is not None:
        shared = torch.cat([shared, registers], 1)
    keys = torch.cat([per_window(take(k, shared)), windows(k[:, :, 1:])], 2)
    values = torch.cat([per_window(take(v, shared)), windows(v[:, :, 1:])], 2)
    out = F.scaled_dot_product_attention(windows(q[:, :, 1:]), keys, values, scale=scale)
    out = out.reshape(batch, count, count, heads, size, size, dim).permute(0, 3, 1, 4, 2, 5, 6)
    out = torch.cat([q.new_zeros(batch, heads, 1, dim), out.reshape(batch, heads, grid * grid, dim)], 2)
    global_out = F.scaled_dot_product_attention(take(q, shared), k, v, scale=scale)
    return out.scatter(2, shared[:, None, :, None].expand(-1, heads, -1, dim), global_out)


def merge_blocks(h, grid, frac):
    """Average the ``frac`` most uniform 2x2 patch blocks into one token each.

    Returns the shorter sequence ``[CLS, merged..., unmerged...]``, an additive
    ``[B, 1, 1, N]`` key bias (log of each token's size, for proportional
    attention) and what :func:`unmerge_blocks` needs to undo it.
    """
    batch, _, dim = h.shape
    half = grid // 2
    blocks = h[:, 1:].reshape(batch, half, 2, half, 2, dim).permute(0, 1, 3, 2, 4, 5)
    blocks = blocks.reshape(batch, half * half, 4, dim)
    with torch.no_grad():
        unit = F.normalize(F.layer_norm(blocks.float(), (dim,)), dim=-1)
        similarity = (unit @ unit.transpose(-1, -2)).sum((-1, -2))
    num_merged = int(round(frac * half * half))
    order = similarity.argsort(dim=1, descending=True)
    merged_idx, kept_idx = order[:, :num_merged], order[:, num_merged:]
    merged = blocks.gather(1, merged_idx[..., None, None].expand(-1, -1, 4, dim)).mean(2)
    kept = blocks.gather(1, kept_idx[..., None, None].expand(-1, -1, 4, dim)).reshape(batch, -1, dim)
    out = torch.cat([h[:, :1], merged, kept], 1)
    bias = torch.zeros(batch, out.shape[1], device=h.device, dtype=h.dtype)
    bias[:, 1:1 + num_merged] = math.log(4)
    return out, bias[:, None, None, :], (merged_idx, kept_idx, num_merged)


def unmerge_blocks(h, grid, info):
    merged_idx, kept_idx, num_merged = info
    batch, _, dim = h.shape
    half = grid // 2
    blocks = torch.empty(batch, half * half, 4, dim, device=h.device, dtype=h.dtype)
    blocks = blocks.scatter(
        1, merged_idx[..., None, None].expand(-1, -1, 4, dim),
        h[:, 1:1 + num_merged, None, :].expand(-1, -1, 4, -1),
    )
    blocks = blocks.scatter(
        1, kept_idx[..., None, None].expand(-1, -1, 4, dim), h[:, 1 + num_merged:].reshape(batch, -1, 4, dim)
    )
    patches = blocks.view(batch, half, half, 2, 2, dim).permute(0, 1, 3, 2, 4, 5).reshape(batch, grid * grid, dim)
    return torch.cat([h[:, :1], patches], 1)


class Encoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, mlp_dim, dropout):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(hidden_size, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        # Toggled by OwlV2.set_gradient_checkpointing; recomputes each block's
        # activations in backward instead of keeping them alive.
        self.gradient_checkpointing = False

    def _should_checkpoint(self, layer, x) -> bool:
        if not (self.gradient_checkpointing and self.training and torch.is_grad_enabled()):
            return False
        # A layer whose input is detached and whose parameters are all frozen
        # stores no activations to begin with, so recomputing it is pure cost.
        # With partial fine-tuning (e.g. only the last N blocks trainable) this
        # skips the frozen prefix of the tower.
        return x.requires_grad or any(p.requires_grad for p in layer.parameters())

    def forward(self, x, mask=None, force=None):
        prune = getattr(self, "prune", None)
        merge = getattr(self, "merge", None)
        extra = aux = merge_info = None
        for index, layer in enumerate(self.layers):
            if merge is not None and index == merge["after"]:
                grid = math.isqrt(x.shape[1] - 1)
                x, bias, merge_info = merge_blocks(x, grid, merge["frac"])
                if merge["prop"]:
                    mask = bias.to(x.dtype if not torch.is_autocast_enabled() else torch.get_autocast_gpu_dtype())
            if prune is not None and index == prune["after"]:
                x, extra, aux = self._prune(x, force, prune)
            window = None
            spec = getattr(self, "window", None)
            if spec is not None and index in spec["layers"]:
                window = (math.isqrt(x.shape[1] - 1), spec["size"], spec["registers"])
            if self._should_checkpoint(layer, x):
                x = checkpoint(layer, x, mask, extra, window, use_reentrant=False)
            else:
                x = layer(x, mask, extra, window)
        if merge_info is not None:
            x = unmerge_blocks(x, grid, merge_info)
        if prune is None:
            return x
        return x, aux

    def _prune(self, x, force, prune):
        """Keep the CLS token and the top-scored ``keep`` of the patch tokens.

        ``force`` ([B, P], training only) lifts those tokens above every score.
        In ``kv`` mode the dropped tokens' current states are returned as
        ``extra``: later blocks still attend to them but never update them.
        """
        scores = self.token_scorer(x[:, 1:]).squeeze(-1).float()
        num_patches = scores.shape[1]
        # ``keep`` is a fraction of the full square grid, so an input with its padding
        # cropped off (fewer tokens) keeps the same token budget, capped at what it has.
        budget = prune.get("full_tokens", num_patches)
        keep = min(num_patches, max(1, round(prune["keep"] * budget)))
        priority = scores.detach()
        if force is not None:
            priority = priority + force.float() * 1e4
        registers = prune.get("registers", 0)
        if registers:
            # High-norm "register" tokens carry global context; never drop them.
            reg = x[:, 1:].norm(dim=-1).topk(registers, dim=1).indices
            priority = priority.scatter_add(1, reg, torch.full_like(reg, 1, dtype=priority.dtype) * 2e4)
        if prune.get("threshold") is not None:
            # Adaptive: every token whose score clears the threshold (registers too),
            # as many as the neediest image of the batch has; ``keep`` is the floor.
            need = int((priority > prune["threshold"]).sum(1).max())
            keep = min(num_patches, max(keep, need))
        order = priority.argsort(dim=1, descending=True)
        top = order[:, :keep].sort(dim=1).values
        gather = torch.cat([torch.zeros_like(top[:, :1]), top + 1], dim=1)
        width = x.shape[-1]
        kept = x.gather(1, gather.unsqueeze(-1).expand(-1, -1, width))
        extra = None
        if prune["mode"] == "kv":
            rest = order[:, keep:] + 1
            extra = x.gather(1, rest.unsqueeze(-1).expand(-1, -1, width))
        return kept, extra, (scores, top)

class VisionTower(nn.Module):
    def __init__(self, hidden_size, patch_size, image_size, num_layers, num_heads, mlp_dim):
        super().__init__()
        
        self.class_embedding = nn.Parameter(torch.randn(hidden_size))
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.num_patches = (image_size // patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, hidden_size)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

        # layer norm
        self.pre_layernorm = nn.LayerNorm(hidden_size,eps=1e-5)
        self.post_layernorm = nn.LayerNorm(hidden_size,eps=1e-5)

        # Encoder
        self.encoder = Encoder(hidden_size, num_layers=num_layers, num_heads=num_heads, mlp_dim=mlp_dim, dropout=0.0)


    def grid_position_embedding(self, rows: int, cols: int) -> torch.Tensor:
        """``[1, 1 + rows * cols, D]`` position embeddings of the top-left ``rows`` x ``cols``
        patches. OWLv2 pads images to a square at the bottom and right, so an input with
        that padding cropped off keeps the positions its patches have in the full grid."""
        positions = self.position_embedding(self.position_ids)
        side = math.isqrt(self.num_patches)
        if (rows, cols) == (side, side):
            return positions
        grid = positions[:, 1:].reshape(1, side, side, -1)[:, :rows, :cols]
        return torch.cat([positions[:, :1], grid.reshape(1, rows * cols, -1)], dim=1)

    def forward(self, x, keep_idx=None, prune_force=None, key_bias=None):
        batch_size = x.shape[0]
        patch_embeds = self.patch_embedding(x)  # shape = [batch_size, num_channels, height, width]
        rows, cols = patch_embeds.shape[-2:]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.grid_position_embedding(rows, cols)
        embeddings = self.pre_layernorm(embeddings)
        if keep_idx is not None:
            # Token drop: only the CLS token and the listed patches enter the encoder.
            gather = torch.cat([torch.zeros_like(keep_idx[:, :1]), keep_idx + 1], dim=1)
            embeddings = embeddings.gather(1, gather.unsqueeze(-1).expand(-1, -1, embeddings.shape[-1]))
        
        mask = None
        if key_bias is not None:
            # [B, 1 + M] additive key bias (token drop with density correction)
            dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else embeddings.dtype
            mask = key_bias[:, None, None, :].to(dtype)
        x = self.encoder(embeddings, mask, force=prune_force) # no mask for vision transformer
        self.prune_aux = None
        if isinstance(x, tuple):
            x, self.prune_aux = x
        pooled_output = x[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return pooled_output, x
class TextTower(nn.Module):
    def __init__(self, hidden_size, num_positions, vocab_size, num_layers, num_heads, mlp_dim):
        super().__init__()
        
        # Embeddings
        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.position_embedding = nn.Embedding(self.num_positions, hidden_size)
        self.token_embedding = nn.Embedding(self.vocab_size, hidden_size)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

        # layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        # Encoder
        self.encoder = Encoder(hidden_size, num_layers=num_layers, num_heads=num_heads, mlp_dim=mlp_dim, dropout=0.0)


    def forward(self, input_ids, attention_mask=None):
        # Embeddings
        seq_length = input_ids.shape[-1]
        position_ids = self.position_ids[:, :seq_length]
        inputs_embeds = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        hidden = inputs_embeds + position_embeddings

        # Build combined causal + padding attention mask on-device (no CPU sync).
        combined_mask = build_causal_padding_mask(
            input_ids, attention_mask=attention_mask, dtype=hidden.dtype
        )

        # Encoder
        encoder_outputs = self.encoder(hidden, combined_mask)
        last_hidden_state = self.final_layer_norm(encoder_outputs)
        # Take features from the end-of-text token (the highest-valued token id
        # in each sequence under CLIP's tokenizer).
        # Casting to torch.int for onnx compatibility: argmax doesn't support
        # int64 inputs with opset 14.
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(torch.int).argmax(dim=-1),
        ]

        return pooled_output, encoder_outputs

class BoxPredictionHead(nn.Module):
    def __init__(self, hidden_size, out_dim: int = 4):
        super().__init__()

        width = hidden_size
        self.dense0 = nn.Linear(width, width)
        self.dense1 = nn.Linear(width, width)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(width, out_dim)

    def forward(self, image_features: torch.Tensor) -> torch.FloatTensor:
        output = self.dense0(image_features)
        output = self.gelu(output)
        output = self.dense1(output)
        output = self.gelu(output)
        output = self.dense2(output)
        return output

class ClassPredictionHead(nn.Module):
    def __init__(self, text_dim, vision_dim):
        super().__init__()

        out_dim = text_dim
        self.query_dim = vision_dim

        self.dense0 = nn.Linear(self.query_dim, out_dim)
        self.logit_shift = nn.Linear(self.query_dim, 1)
        self.logit_scale = nn.Linear(self.query_dim, 1)
        self.elu = nn.ELU()

    def forward(
        self,
        image_embeds: torch.FloatTensor,
        query_embeds: Optional[torch.FloatTensor],
        query_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.FloatTensor]:
        image_class_embeds = self.dense0(image_embeds)
        if query_embeds is None:
            device = image_class_embeds.device
            batch_size, num_patches = image_class_embeds.shape[:2]
            pred_logits = torch.zeros((batch_size, num_patches, self.query_dim)).to(device)
            return (pred_logits, image_class_embeds)

        # Normalize image and text features
        image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
        query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)

        # Get class predictions
        pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)
        
        # Apply a learnable shift and scale to logits
        logit_shift = self.logit_shift(image_embeds) 
        logit_scale = self.logit_scale(image_embeds)
        logit_scale = self.elu(logit_scale) + 1
        pred_logits = (pred_logits + logit_shift) * logit_scale

        if query_mask is not None:
            if query_mask.ndim > 1:
                query_mask = torch.unsqueeze(query_mask, dim=-2)

            pred_logits = torch.where(
                query_mask == 0, torch.finfo(pred_logits.dtype).min, pred_logits
            )
        return (pred_logits, image_class_embeds)


# Bumped whenever any module registers a parameter or submodule, so cached
# parameter lists (OwlV2.parameter_state) notice replaced weights.
_parameter_generation = 0


def _bump_parameter_generation(*args):
    global _parameter_generation
    _parameter_generation += 1


nn.modules.module.register_module_parameter_registration_hook(_bump_parameter_generation)
nn.modules.module.register_module_module_registration_hook(_bump_parameter_generation)


def _same_tensor(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> bool:
    if a is None or b is None:
        return a is b
    return a.shape == b.shape and a.device == b.device and bool(torch.equal(a, b))


def _watch(*tensors) -> Optional[tuple]:
    """The tensors with their version counters, for :func:`_unchanged`.

    None when one is an inference tensor: those keep no version counter, so
    in-place writes to them cannot be seen.
    """
    if any(t is not None and t.is_inference() for t in tensors):
        return None
    return tuple((t, None if t is None else t._version) for t in tensors)


def _unchanged(watched: Optional[tuple], *tensors) -> bool:
    """Whether ``tensors`` are the very ones :func:`_watch` saw, not written to since.

    Unlike comparing values, this does not wait for queued GPU work.
    """
    return watched is not None and len(watched) == len(tensors) and all(
        seen is t and (t is None or t._version == version) for (seen, version), t in zip(watched, tensors)
    )


class OwlV2(nn.Module):
    def __init__(self, model_type="base"):
        super().__init__()

        if model_type == "base":
            self.model_string = "google/owlv2-base-patch16-ensemble"
            project_dim=512
            vision_dim=768 
            text_dim=512
            image_size=960
            patch_size=16
            logit_scale = 2.6592
            mlp_dim = 3072
            num_layers = 12
            num_text_layers = 12
            num_heads = 12

            text_num_heads = 8
            text_mlp_dim = 2048
            
        else:
            self.model_string = "google/owlv2-large-patch14-ensemble"
            project_dim=768
            vision_dim=1024
            text_dim=768
            image_size=1008
            patch_size=14
            logit_scale = 2.6592
            mlp_dim = 4096
            num_layers = 24
            num_text_layers = 12
            num_heads = 16
            text_num_heads = 12
            text_mlp_dim = 3072
            
        
        self.projected_dim = project_dim
        self.vision_dim = vision_dim
        self.text_dim = text_dim

        self.image_size = image_size
        self.patch_size = patch_size

        self.vision_model = VisionTower(hidden_size=self.vision_dim, patch_size=self.patch_size, image_size=image_size, num_layers=num_layers, num_heads=num_heads, mlp_dim=mlp_dim)
        self.text_model = TextTower(hidden_size=self.text_dim, vocab_size=49408, num_positions=16, num_layers=num_text_layers, num_heads=text_num_heads, mlp_dim=text_mlp_dim)

        self.visual_projection = nn.Linear(self.vision_dim, self.projected_dim, bias=False)
        self.text_projection = nn.Linear(self.text_dim, self.projected_dim, bias=False)

        self.logit_scale = nn.Parameter(torch.tensor(logit_scale))
        self.layer_norm = nn.LayerNorm(self.vision_dim, eps=1e-5)


        self.class_head = ClassPredictionHead(self.text_dim, self.vision_dim)
        self.box_head = BoxPredictionHead(self.vision_dim)
        self.objectness_head = BoxPredictionHead(self.vision_dim, out_dim=1)

        self.sqrt_num_patches = self.image_size // self.patch_size
        # Register as a buffer so it follows .to(device)/.cuda() without
        # re-materialising on every forward.
        self.register_buffer(
            "box_bias", self.compute_box_bias(self.sqrt_num_patches), persistent=False
        )
        # One prediction per patch token unless enable_point_slots() adds more.
        self.point_slot_grid = 1
        self.point_slot_head = None
        # Same result as the accurate path below, computed in torch.
        self.image_transform_fast = T.Compose([
            TorchScipyResize(self.image_size),
            T.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
        ])
        # The scipy reference (HF's Owlv2ImageProcessor), kept to check against.
        self.image_transform_accurate = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            SquarePad(),
            ScipyResize(self.image_size),
            T.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
        ])
        self.image_transform = self.image_transform_fast

        self.image_transform_unnormed = TorchScipyResize(self.image_size)

        self.owlv2_img_normalize = T.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD)

        # Inference-only options of preprocess_image/postprocess_boxes (set_preprocessing);
        # the defaults are HF's pipeline.
        self.preprocess_blur = 0.0
        self.preprocess_long_side = None
        self.preprocess_pad_drop = False
        # Full-square-grid index of the tokens in the last detection's outputs, or
        # None when they are the full grid in order; see
        # forward_object_detection_from_embeddings.
        self.detection_token_index = None
        self._query_cache = None
        self._parameter_lists = {}

        self._load_model(self.model_string)

    def _load_model(self, model_path):
        state_dict = {}
        cache_path = find_safetensors_in_cache(model_path)

        if len(cache_path) == 0:
            hf_hub_download(repo_id=model_path, filename="model.safetensors")
            cache_path = find_safetensors_in_cache(model_path)
        if len(cache_path) == 0:
            raise FileNotFoundError(
                f"Could not locate safetensors weights for {model_path} after download."
            )
        if len(cache_path) > 1:
            raise RuntimeError(
                f"Ambiguous cache: found {len(cache_path)} safetensors candidates for "
                f"{model_path}: {cache_path}"
            )
        with safe_open(cache_path[0], framework="pt") as f:
            for k in f.keys():
                tens = f.get_tensor(k)
                k = k.replace("owlv2.","").replace(".embeddings","")
                k = k.replace("mlp.fc1","mlp.0").replace("mlp.fc2","mlp.2")
                state_dict[k] = tens

        # Construction happens before optimizers can hold parameter references, so
        # assigning the safetensors-backed tensors avoids copying the full checkpoint
        # into the already allocated module parameters.
        self.load_state_dict(state_dict, strict=True, assign=True)
    
    def set_gradient_checkpointing(
        self, enabled: bool = True, vision: bool = True, text: bool = True
    ) -> "OwlV2":
        """Trade compute for activation memory in the transformer towers.

        Each selected encoder block is re-run during backward instead of keeping
        its activations, which is where nearly all of the training memory sits.
        Checkpointing only takes effect while the module is in training mode with
        grad enabled, so inference and evaluation paths are untouched.

        Args:
            enabled: turn checkpointing on or off.
            vision: apply to the vision tower's encoder.
            text: apply to the text tower's encoder.
        """
        if vision:
            self.vision_model.encoder.gradient_checkpointing = enabled
        if text:
            self.text_model.encoder.gradient_checkpointing = enabled
        return self

    @property
    def gradient_checkpointing(self) -> bool:
        """True if either tower is checkpointing."""
        return (
            self.vision_model.encoder.gradient_checkpointing
            or self.text_model.encoder.gradient_checkpointing
        )

    def preprocess_image(self, image, fast=True, device=None):
        """Preprocess one or more images for OwlV2 detection.

        Accepts a single PIL image / file path, or a list/tuple of either.
        Always returns a ``[B, 3, H, W]`` float tensor (B=1 for a single input).

        Both paths give the same pixels (within ~1e-6); ``fast=False`` runs the
        original scipy implementation, which is several times slower on the CPU.
        The fast path applies resize, blur, crop and normalisation as one cached
        operator per image size (see ``_preprocess_operators``). Images are
        preprocessed one at a time (their sizes differ) and stacked into a batch
        so downstream forward passes can run with B>1.

        ``device`` (fast path only) moves each uint8 image there and preprocesses
        it on that device, returning the batch on it. On CUDA that is several
        times faster than the CPU, which matters when the CPU resize would
        otherwise leave the GPU idle. ``None`` keeps it on the CPU.
        """
        if device is not None and not fast:
            raise ValueError("device needs fast=True; the scipy path runs on the CPU only")

        images = list(image) if isinstance(image, (list, tuple)) else [image]
        images = [Image.open(img) if isinstance(img, str) else img for img in images]
        side = self.preprocess_long_side or self.image_size
        rows = cols = side
        if self.preprocess_pad_drop:
            # Drop the patches that are all padding; the batch shares its largest grid.
            sizes = [(img.height, img.width) if isinstance(img, Image.Image) else tuple(img.shape[-2:])
                     for img in images]
            rows, cols = (n * self.patch_size for n in self.padding_free_grid(sizes))
        if fast:
            # Keep pixels float32 under a caller's autocast, which would run the
            # sparse operators in float16 (~3e-3 off instead of ~1e-6).
            with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
                return torch.stack([self._preprocess_fast(img, device, side, rows, cols) for img in images])
        pixel_values = torch.stack([self._preprocess_one(img) for img in images])
        return pixel_values[..., :rows, :cols] if self.preprocess_pad_drop else pixel_values

    def set_preprocessing(self, blur: float = 0.0, long_side: Optional[int] = None,
                          pad_drop: bool = False) -> "OwlV2":
        """Inference preprocessing that departs from HF's, for the original weights.

        Applies to :meth:`preprocess_image` and :meth:`postprocess_boxes` only
        (training transforms are untouched); call with no arguments to reset.

        - ``blur``: Gaussian sigma in pixels of the resized image. OWLv2 wants
          slightly soft input: with base, 0.5 gave +0.011 COCO val mAP zero-shot
          and +0.018 on Fashionpedia, nothing on DIOR. With large only +0.002
          COCO, and Fashionpedia loses 0.006-0.012, so leave it off there.
        - ``long_side``: resize the long side to this (a multiple of the patch
          size, at most ``image_size``) and place it in the top-left of the
          ``image_size`` canvas, keeping the original position embeddings and box
          priors. Implies ``pad_drop``, so the tower sees fewer tokens. With
          blur, 768 runs 1.6x faster than 960 and costs COCO val 0.007 mAP (still
          +0.007 over HF's pipeline), Fashionpedia 0.003-0.011, DIOR nothing. Unlike
          :meth:`set_image_size`, which resamples the position embeddings and
          loses 0.05-0.2 mAP zero-shot.
        - ``pad_drop``: crop the patches that are all square padding (free in
          accuracy; see :meth:`padding_free_grid`).

        ``OwlV2TRT`` runs cropped inputs on its padding-free engine.
        """
        if long_side is not None:
            if long_side % self.patch_size or not 0 < long_side <= self.image_size:
                raise ValueError(f"long_side must be a multiple of {self.patch_size} "
                                 f"up to {self.image_size}, got {long_side}")
            if long_side < self.image_size and self.point_slot_head is not None:
                raise ValueError("Point slots need the full square grid")
            pad_drop = True
        if pad_drop and self.point_slot_head is not None:
            raise ValueError("Point slots need the full square grid; do not crop the padding")
        self.preprocess_blur = float(blur)
        self.preprocess_long_side = None if long_side == self.image_size else long_side
        self.preprocess_pad_drop = pad_drop
        return self

    def _preprocess_fast(self, image, device, side, rows, cols):
        """One image (PIL or ``(C, H, W)`` tensor) to the normalised ``(3, rows, cols)``
        top-left crop of its ``side`` canvas."""
        if isinstance(image, Image.Image):
            # np.asarray is 5x cheaper than np.array but read-only; nothing
            # writes to it, so torch's warning doesn't apply.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                hwc = torch.from_numpy(np.asarray(image))
            if hwc.ndim == 2:
                hwc = hwc.unsqueeze(-1)
        else:
            hwc = image.as_subclass(torch.Tensor).permute(1, 2, 0)
        if device is not None:
            hwc = hwc.to(device)
        h, w, c = hwc.shape
        scale = 1.0 if hwc.is_floating_point() else 1.0 / torch.iinfo(hwc.dtype).max
        ay, ax, fill, inv_std, shift = _preprocess_operators(
            h, w, side, rows, cols, self.preprocess_blur, scale, hwc.device
        )
        # Contract rows on the (H, W*C) view, then columns on the (W, rows*C) view.
        y = torch.sparse.mm(ay, hwc.reshape(h, w * c).float())
        y = y.view(rows, w, c).transpose(0, 1).reshape(w, rows * c)
        out = torch.sparse.mm(ax, y).view(cols, rows, c).add_(fill)
        return torch.addcmul(shift, out, inv_std).permute(2, 1, 0)

    def _preprocess_one(self, image):
        """One image (PIL or ``(C, H, W)`` tensor) to a normalised square canvas, with scipy."""
        long_side = self.preprocess_long_side
        if long_side is None:
            pixel_values = self.image_transform_accurate(image)
        else:
            pixel_values = self.owlv2_img_normalize(ScipyResize(long_side)(SquarePad()(
                TF.to_dtype(TF.to_image(image), torch.float32, scale=True))))
        if self.preprocess_blur > 0:
            radius = max(1, int(3 * self.preprocess_blur + 0.5))
            pixel_values = TF.gaussian_blur(pixel_values, [2 * radius + 1] * 2, [self.preprocess_blur] * 2)
        return pixel_values


    def _get_vision_outputs(self, pixel_values):
        """Return pooled and dense vision outputs without projecting either one."""
        return self.vision_model(pixel_values)

    def get_vision_features(self, pixel_values, normalize=True):
        vision_pooled, vision_full = self._get_vision_outputs(pixel_values)
        vision_features = self.visual_projection(vision_pooled)
        if normalize:
            vision_features = vision_features / (
                torch.linalg.norm(vision_features, dim=-1, keepdim=True) + 1e-6
            )
        return vision_features, vision_pooled, vision_full

    def get_text_features(self, token_ids, attention_mask, normalize=True):
        y, _ = self.text_model(token_ids, attention_mask)
        y = self.text_projection(y)
        if normalize:
            y = y / (torch.linalg.norm(y, dim=-1, keepdim=True) + 1e-6)
        return y

    def encode_detection_queries(self, token_ids, attention_mask=None):
        """Encode a shared 2D query set once for reuse across image batches.

        Returns projected text embeddings ``[num_queries, text_dim]`` and a boolean
        query-presence mask ``[num_queries]``. The embeddings can be passed to
        :meth:`forward_object_detection_from_embeddings` repeatedly.
        """
        if token_ids.ndim != 2:
            raise ValueError(
                "encode_detection_queries expects token_ids with shape "
                f"[num_queries, seq_len], got {tuple(token_ids.shape)}"
            )
        if attention_mask is not None and attention_mask.shape != token_ids.shape:
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} does not match "
                f"token_ids shape {tuple(token_ids.shape)}"
            )
        return self.get_text_features(token_ids, attention_mask), token_ids[..., 0] > 0

    def parameter_state(self, key, parameters) -> tuple:
        """``(id, data_ptr, version)`` of each parameter ``parameters()`` yields.

        It changes whenever one is replaced, moved or written in place, so
        comparing it tells whether weights changed since an engine or a cache was
        made. Walking the modules costs ~0.15 ms on base, so the list is kept per
        ``key`` until a parameter or submodule is registered anywhere.
        """
        cached = self._parameter_lists.get(key)
        if cached is None or cached[0] != _parameter_generation:
            cached = (_parameter_generation, list(parameters()))
            self._parameter_lists[key] = cached
        return tuple((id(p), p.data_ptr(), p._version) for p in cached[1])

    def _detection_query_features(self, token_ids, attention_mask):
        """:meth:`get_text_features`, reusing the last result for the same queries.

        Detection usually repeats one query set image after image; re-encoding it
        costs ~4 ms on base. The cache is skipped under autograd and dropped when
        the text weights or the autocast dtype change. Passing the same unmodified
        tensors again hits without comparing their values, whose device sync held
        back the next launch (~0.1 ms per image at 768 on a 5070 Ti).
        """
        if torch.is_grad_enabled():
            self._query_cache = None
            return self.get_text_features(token_ids, attention_mask)
        device_type = token_ids.device.type
        autocast = torch.is_autocast_enabled(device_type) and torch.get_autocast_dtype(device_type)
        state = (autocast, self.parameter_state(
            "text", lambda: (p for m in (self.text_model, self.text_projection) for p in m.parameters())
        ))
        cache = self._query_cache
        if cache is not None and cache[0] == state and (
            _unchanged(cache[4], token_ids, attention_mask)
            or _same_tensor(cache[1], token_ids) and _same_tensor(cache[2], attention_mask)
        ):
            return cache[3]
        features = self.get_text_features(token_ids, attention_mask)
        mask = None if attention_mask is None else attention_mask.clone()
        self._query_cache = (state, token_ids.clone(), mask, features, _watch(token_ids, attention_mask))
        return features

    def forward(self, pixel_values, token_ids, attention_mask):
        vision_features, vision_pooled, vision_full = self.get_vision_features(pixel_values)
        text_features = self.get_text_features(token_ids, attention_mask)

        logit_scale = self.logit_scale.exp().to(device=vision_features.device)
        logits_per_image = torch.matmul(vision_features, text_features.t()) * logit_scale
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text, vision_features, text_features, vision_full

    def forward_object_detection(self, pixel_values, token_ids, attention_mask=None, keep_idx=None, prune_force=None,
                                 key_bias=None):
        batch_size = pixel_values.shape[0]
        shared_queries = token_ids.ndim == 2
        token_ids, attention_mask, max_text_queries = normalize_detection_queries(
            token_ids,
            attention_mask,
            batch_size,
            repeat_shared=False,
        )
        text_features = self._detection_query_features(token_ids, attention_mask)

        if shared_queries:
            query_mask = token_ids[..., 0] > 0
        else:
            text_features = text_features.reshape(
                batch_size, max_text_queries, text_features.shape[-1]
            )
            token_ids = token_ids.reshape(
                batch_size, max_text_queries, token_ids.shape[-1]
            )
            query_mask = token_ids[..., 0] > 0

        return self.forward_object_detection_from_embeddings(
            pixel_values, text_features, query_mask, keep_idx=keep_idx, prune_force=prune_force,
            key_bias=key_bias,
        )

    def forward_object_detection_from_embeddings(
        self,
        pixel_values: torch.Tensor,
        query_embeds: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        keep_idx: Optional[torch.Tensor] = None,
        prune_force: Optional[torch.Tensor] = None,
        key_bias: Optional[torch.Tensor] = None,
    ):
        """Run detection with projected text embeddings.

        ``query_embeds`` may be shared ``[num_queries, text_dim]`` or explicitly
        batched ``[batch_size, num_queries, text_dim]``. Shared embeddings are
        expanded as a view, so identical text queries are never re-encoded for each
        image.

        Outputs follow the tokens that reach the heads. When those are not the
        full square grid in order (padding cropped off, or a pruned tower),
        ``self.detection_token_index`` holds each one's index in the full square
        grid, ``[batch_size, num_tokens]``; otherwise it is None.
        """
        batch_size = pixel_values.shape[0]
        if query_embeds.ndim == 2:
            if query_embeds.shape[-1] != self.text_dim:
                raise ValueError(
                    f"Expected query embedding dim {self.text_dim}, got "
                    f"{query_embeds.shape[-1]}"
                )
            num_queries = query_embeds.shape[0]
            text_features = query_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            if query_mask is not None:
                if query_mask.shape != (num_queries,):
                    raise ValueError(
                        "A shared query_mask must have shape "
                        f"[{num_queries}], got {tuple(query_mask.shape)}"
                    )
                query_mask = query_mask.unsqueeze(0).expand(batch_size, -1)
        elif query_embeds.ndim == 3:
            if query_embeds.shape[0] != batch_size:
                raise ValueError(
                    f"Query embedding batch {query_embeds.shape[0]} does not match "
                    f"pixel batch {batch_size}"
                )
            if query_embeds.shape[-1] != self.text_dim:
                raise ValueError(
                    f"Expected query embedding dim {self.text_dim}, got "
                    f"{query_embeds.shape[-1]}"
                )
            text_features = query_embeds
            if query_mask is not None and query_mask.shape != query_embeds.shape[:2]:
                raise ValueError(
                    f"query_mask must match query batch shape {query_embeds.shape[:2]}, "
                    f"got {tuple(query_mask.shape)}"
                )
        else:
            raise ValueError(
                "query_embeds must have shape [num_queries, text_dim] or "
                "[batch_size, num_queries, text_dim], got "
                f"{tuple(query_embeds.shape)}"
            )

        if text_features.device != pixel_values.device:
            raise ValueError(
                f"query_embeds are on {text_features.device}, but pixel_values are "
                f"on {pixel_values.device}"
            )
        if query_mask is not None:
            query_mask = query_mask.to(device=text_features.device, dtype=torch.bool)

        # Detection needs dense vision states, not the pooled visual projection.
        # A smaller, non-square input is a padding-free crop (see padding_free_grid).
        grid = (pixel_values.shape[-2] // self.patch_size, pixel_values.shape[-1] // self.patch_size)
        pruning = getattr(self.vision_model.encoder, "prune", None) is not None
        if keep_idx is not None or pruning:
            # Predictions for the kept tokens only, in keep_idx order (slots token-major).
            _, vision_full = self.vision_model(pixel_values, keep_idx, prune_force, key_bias)
            if pruning:
                keep_idx = self.vision_model.prune_aux[1]
            self.detection_token_index = (
                self.full_grid_index(keep_idx, grid[1]) if self.point_slot_head is None else None
            )
            feature_map = self.vision_model.post_layernorm(vision_full)
            image_feats = self.layer_norm(feature_map[:, 1:, :] * feature_map[:, :1, :])
            image_feats = self.point_slot_features(image_feats, kept=True)
            box_bias = self.kept_box_bias(keep_idx, grid[1])
        else:
            _, vision_full = self._get_vision_outputs(pixel_values)
            image_feats, feature_map = self._detection_image_features_from_vision(vision_full, grid)
            self.detection_token_index = None
            if grid == (self.sqrt_num_patches,) * 2:
                image_feats = self.point_slot_features(image_feats)
                box_bias = self.detection_box_bias
            elif self.point_slot_head is not None:
                raise ValueError("Point slots need the full square grid; do not crop the padding")
            else:
                box_bias = self.grid_box_bias(*grid)
                local = torch.arange(grid[0] * grid[1], device=pixel_values.device)
                self.detection_token_index = self.full_grid_index(local, grid[1]).expand(batch_size, -1)
        if text_features.dtype != image_feats.dtype:
            text_features = text_features.to(dtype=image_feats.dtype)

        # Predict object classes [batch_size, num_patches, num_queries+1]
        (pred_logits, class_embeds) = self.class_head(image_feats, text_features, query_mask)

        # Predict objectness
        objectness_logits = self.objectness_head(image_feats)
        objectness_logits = objectness_logits[..., 0]

        # Predict object boxes. Use functional add (not +=) so autograd is
        # happy during training.
        pred_boxes = self.box_head(image_feats)
        pred_boxes = torch.sigmoid(pred_boxes + box_bias)

        return pred_logits, objectness_logits, pred_boxes, class_embeds, (feature_map, text_features)

    def set_image_size(self, image_size: int) -> "OwlV2":
        """Run at another input size: resample the position grid (bicubic) and rebuild
        the box priors and image transforms. Call before :meth:`enable_point_slots`."""
        if self.point_slot_head is not None:
            raise RuntimeError("Set the image size before enabling point slots")
        old_grid, grid = self.sqrt_num_patches, image_size // self.patch_size
        if grid * self.patch_size != image_size:
            raise ValueError(f"image_size must be a multiple of {self.patch_size}")
        vision = self.vision_model
        weight = vision.position_embedding.weight.detach()
        patches = weight[1:].T.reshape(1, -1, old_grid, old_grid).float()
        patches = F.interpolate(patches, size=(grid, grid), mode="bicubic", align_corners=False)
        patches = patches.reshape(weight.shape[1], grid * grid).T.to(weight.dtype)
        vision.position_embedding = nn.Embedding(grid * grid + 1, weight.shape[1]).to(weight.device, weight.dtype)
        with torch.no_grad():
            vision.position_embedding.weight.copy_(torch.cat([weight[:1], patches], 0))
        vision.num_patches, vision.num_positions = grid * grid, grid * grid + 1
        vision.register_buffer(
            "position_ids", torch.arange(grid * grid + 1, device=weight.device).expand((1, -1)), persistent=False
        )
        self.image_size, self.sqrt_num_patches = image_size, grid
        self.register_buffer("box_bias", self.compute_box_bias(grid).to(weight.device), persistent=False)
        self.image_transform_fast = T.Compose([
            TorchScipyResize(image_size), T.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD)
        ])
        self.image_transform = self.image_transform_fast
        self.image_transform_unnormed = TorchScipyResize(image_size)
        return self

    def enable_window_attention(self, layers, size: int, registers: int = 16) -> "OwlV2":
        """Windowed attention in ``layers``: ``size`` x ``size`` patch windows, plus the
        CLS token and the ``registers`` highest-norm patch tokens, which stay global."""
        if self.sqrt_num_patches % size:
            raise ValueError(f"window {size} does not tile the {self.sqrt_num_patches} grid")
        self.vision_model.encoder.window = {"layers": set(layers), "size": size, "registers": registers}
        return self

    def enable_token_merging(self, after: int, frac: float, prop: bool = True) -> "OwlV2":
        """From block ``after`` on, the ``frac`` most uniform 2x2 patch blocks run as one
        token each; their output is copied back to all four positions at the end."""
        self.vision_model.encoder.merge = {"after": after, "frac": frac, "prop": prop}
        return self

    def enable_token_pruning(self, after: int, keep: float, mode: str = "drop", registers: int = 0,
                             threshold: Optional[float] = None) -> "OwlV2":
        """From block ``after`` on, update only the top-scored ``keep`` of the patch tokens.

        A small MLP on the residual stream scores the tokens; detections come
        from the kept tokens only. ``keep`` is a fraction of the full grid.
        With ``threshold``, every token scoring above it is kept too (``keep``
        becomes the floor; a batch keeps as many as its neediest image).
        ``mode="kv"`` lets the dropped tokens' frozen states still serve as
        keys/values in the later blocks.
        """
        if mode not in ("drop", "kv"):
            raise ValueError(f"mode must be drop or kv, got {mode!r}")
        encoder = self.vision_model.encoder
        dim = self.vision_dim
        device, dtype = self.box_bias.device, self.layer_norm.weight.dtype
        encoder.token_scorer = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, 1)
        ).to(device=device, dtype=dtype)
        encoder.prune = {
            "after": after, "keep": keep, "mode": mode, "registers": registers,
            "threshold": threshold, "full_tokens": self.sqrt_num_patches ** 2,
        }
        return self

    def load_token_scorer(self, path, keep: Optional[float] = None, threshold: Optional[float] = None,
                          registers: Optional[int] = None) -> "OwlV2":
        """Prune with a scorer saved by ``tools/arch_speedup/distill_scorer.py``.

        Such a scorer is distilled from this model's own objectness on unlabelled
        images, so the tower weights stay the original ones. ``keep``,
        ``threshold`` and ``registers`` override the values stored with it.
        """
        saved = torch.load(path, map_location="cpu", weights_only=True)
        config = saved["config"]
        self.enable_token_pruning(
            config["after"],
            config["keep"] if keep is None else keep,
            registers=config["registers"] if registers is None else registers,
            threshold=config.get("threshold") if threshold is None else threshold,
        )
        self.vision_model.encoder.token_scorer.load_state_dict(saved["scorer"])
        return self

    def full_grid_index(self, index: torch.Tensor, cols: int) -> torch.Tensor:
        """Patch ``index`` in a grid ``cols`` wide (padding cropped) -> index in the full square grid."""
        P = self.sqrt_num_patches
        return index if cols == P else index // cols * P + index % cols

    def kept_box_bias(self, keep_idx: torch.Tensor, cols: Optional[int] = None) -> torch.Tensor:
        """``[B, M * grid**2, 4]`` box bias for the slots of the kept patch tokens.

        ``keep_idx`` indexes the input's patch grid; ``cols`` is its width when the
        padding was cropped off (the full grid otherwise).
        """
        S, P = self.point_slot_grid, self.sqrt_num_patches
        if cols is not None:
            keep_idx = self.full_grid_index(keep_idx, cols)
        if S == 1:
            return self.box_bias[keep_idx]
        r, c = keep_idx // P, keep_idx % P
        sub = torch.arange(S, device=keep_idx.device)
        rows = (r[..., None, None] * S + sub[:, None]) * (P * S)
        cols = c[..., None, None] * S + sub[None, :]
        slot_index = (rows + cols).flatten(-2)  # [B, M, S*S], (sy, sx) row-major
        return self.point_slot_box_bias[slot_index.flatten(1)]

    def enable_point_slots(self, grid: int) -> "OwlV2":
        """Give every patch token ``grid x grid`` point slots instead of one prediction.

        A linear layer maps each token's detection embedding to ``grid**2`` slot
        embeddings, which the unchanged class, objectness and box heads score as
        ordinary queries. The slots are laid out as a row-major ``P*grid`` square
        grid, so this is a stride ``patch_size / grid`` detector on the same
        backbone. The layer starts as the identity tiled ``grid**2`` times, so
        every slot begins as its parent token. Each slot's centre prior is the
        parent's, shifted to the centre of its sub-cell (symmetrically, so the
        slots straddle what the pretrained head predicts). The size prior stays
        at one parent patch.
        """
        if grid < 1:
            raise ValueError(f"grid must be at least 1, got {grid}")
        if self.point_slot_head is not None:
            raise RuntimeError(f"Point slots are already enabled ({self.point_slot_grid}x{self.point_slot_grid})")
        if grid == 1:
            return self
        device, dtype = self.box_bias.device, self.layer_norm.weight.dtype
        dim = self.vision_dim
        head = nn.Linear(dim, grid * grid * dim, device=device, dtype=dtype)
        with torch.no_grad():
            head.weight.copy_(torch.eye(dim, device=device, dtype=dtype).repeat(grid * grid, 1))
            head.bias.zero_()
        self.point_slot_head = head
        self.point_slot_grid = grid
        self.register_buffer(
            "point_slot_box_bias", self.compute_point_slot_box_bias(grid).to(device), persistent=False
        )
        return self

    def compute_point_slot_box_bias(self, grid: int) -> torch.Tensor:
        """Box bias for :meth:`enable_point_slots`, ``[(P*grid)**2, 4]`` in slot order."""
        P = self.sqrt_num_patches
        # (y, sub_y, x, sub_x) is row-major order on the fine grid.
        coords = self.normalize_grid_corner_coordinates(P).view(P, 1, P, 1, 2).repeat(1, grid, 1, grid, 1)
        offsets = (torch.arange(grid, dtype=torch.float32) + 0.5) / (P * grid) - 0.5 / P
        coords[..., 0] += offsets.view(1, 1, 1, grid)
        coords[..., 1] += offsets.view(1, grid, 1, 1)
        coords = torch.clip(coords.view(-1, 2), 0.0, 1.0)
        coord_bias = torch.log(coords + 1e-4) - torch.log1p(-coords + 1e-4)
        size_bias = self.compute_box_bias(P)[:1, 2:].expand(coord_bias.shape[0], 2)
        return torch.cat([coord_bias, size_bias], dim=-1)

    def point_slot_features(self, image_feats: torch.Tensor, kept: bool = False) -> torch.Tensor:
        """``[B, P*P, D]`` token features to ``[B, (P*grid)**2, D]`` slot features.

        Returns ``image_feats`` unchanged when point slots are not enabled.
        """
        if self.point_slot_head is None:
            return image_feats
        batch_size, num_tokens, dim = image_feats.shape
        P, S = self.sqrt_num_patches, self.point_slot_grid
        if kept:
            return self.point_slot_head(image_feats).view(batch_size, num_tokens * S * S, dim)
        slots = self.point_slot_head(image_feats).view(batch_size, P, P, S, S, dim)
        return slots.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, (P * S) ** 2, dim)

    @property
    def detection_box_bias(self) -> torch.Tensor:
        """The box bias matching :meth:`point_slot_features`' query layout."""
        return self.box_bias if self.point_slot_head is None else self.point_slot_box_bias

    @staticmethod
    def normalize_grid_corner_coordinates(num_patches: int) -> torch.Tensor:
        # Create grid coordinates using torch
        x_coordinates = torch.arange(1, num_patches + 1, dtype=torch.float32)
        y_coordinates = torch.arange(1, num_patches + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x_coordinates, y_coordinates, indexing="xy")

        # Stack the coordinates and divide by num_patches
        box_coordinates = torch.stack((xx, yy), dim=-1)
        box_coordinates /= num_patches

        # Flatten (h, w, 2) -> (h*w, 2)
        box_coordinates = box_coordinates.view(-1, 2)

        return box_coordinates

    def compute_box_bias(self, num_patches: int, feature_map: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        if feature_map is not None:
            raise ValueError("feature_map has been deprecated as an input. Please pass in num_patches instead")
        # The box center is biased to its position on the feature grid
        box_coordinates = self.normalize_grid_corner_coordinates(num_patches)
        box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)

        # Unnormalize xy
        box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(-box_coordinates + 1e-4)

        # The box size is biased to the patch size
        box_size = torch.full_like(box_coord_bias, 1.0 / num_patches)
        box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)

        # Compute box bias
        box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
        return box_bias


    def postprocess_boxes(self, boxes, image_size, compensate_padding=True):
        if isinstance(image_size, (list, tuple)):
            target_sizes = torch.as_tensor(
                image_size, device=boxes.device, dtype=boxes.dtype
            )
        elif isinstance(image_size, torch.Tensor):
            target_sizes = image_size.to(device=boxes.device, dtype=boxes.dtype)
        else:
            raise ValueError("`target_sizes` must be a list, tuple or torch.Tensor")

        if target_sizes.ndim != 2 or target_sizes.shape[1] != 2:
            raise ValueError(
                "`target_sizes` must have shape [batch_size, 2] as (height, width), "
                f"got {tuple(target_sizes.shape)}"
            )
        if target_sizes.shape[0] != boxes.shape[0]:
            raise ValueError(
                f"target size batch {target_sizes.shape[0]} does not match box batch "
                f"{boxes.shape[0]}"
            )
        image_height, image_width = target_sizes.unbind(1)
        if self.preprocess_long_side is not None:
            # Boxes are normalised to the image_size canvas; the image spans long_side of it.
            boxes = boxes * (self.image_size / self.preprocess_long_side)

        center_x, center_y, width, height = boxes.unbind(-1)
        x_coords = torch.stack(
            (center_x - 0.5 * width, center_x + 0.5 * width), dim=-1
        )
        y_coords = torch.stack(
            (center_y - 0.5 * height, center_y + 0.5 * height), dim=-1
        )
    
        if compensate_padding:
            eps = torch.finfo(boxes.dtype).eps
            height_larger = (image_height > image_width).view(-1, 1, 1)
            width_larger = (image_width > image_height).view(-1, 1, 1)

            x_factor = (image_width / image_height.clamp_min(eps)).view(-1, 1, 1)
            y_factor = (image_height / image_width.clamp_min(eps)).view(-1, 1, 1)
            x_factor = x_factor.clamp_min(eps)
            y_factor = y_factor.clamp_min(eps)

            compensated_x = torch.minimum(x_coords.clamp_min(0), x_factor) / x_factor
            compensated_y = torch.minimum(y_coords.clamp_min(0), y_factor) / y_factor
            x_coords = torch.where(height_larger, compensated_x, x_coords)
            y_coords = torch.where(width_larger, compensated_y, y_coords)

        boxes_xyxy = torch.stack(
            (x_coords[..., 0], y_coords[..., 0], x_coords[..., 1], y_coords[..., 1]),
            dim=-1,
        )
        scale_factor = torch.stack(
            (image_width, image_height, image_width, image_height), dim=1
        ).unsqueeze(1)
        return boxes_xyxy * scale_factor

    @staticmethod
    def select_detections(pred_logits, objectness_logits, top_k: int = 300, multi_label: bool = False):
        """Best ``top_k`` detections per image as ``(scores, labels, token_index)``, each ``[B, K]``.

        By default each token is one detection of its best class, scored
        ``sigmoid(class) * sigmoid(objectness)``. ``multi_label`` ranks every
        (token, class) pair by ``sigmoid(class) * sqrt(sigmoid(objectness))``
        instead, so a token can report several classes: zero-shot +0.004 COCO and
        +0.032 Fashionpedia mAP, but -0.002 on DIOR (keep it off for aerial).
        Take the boxes with ``boxes.gather(1, token_index[..., None].expand(-1, -1, 4))``.
        """
        cls = pred_logits.float().sigmoid()
        obj = objectness_logits.float().sigmoid().unsqueeze(-1)
        if multi_label:
            scores = (cls * obj.sqrt()).flatten(1)
            scores, index = scores.topk(min(top_k, scores.shape[1]), dim=1)
            return scores, index % cls.shape[-1], index // cls.shape[-1]
        scores, labels = (cls * obj).max(-1)
        scores, index = scores.topk(min(top_k, scores.shape[1]), dim=1)
        return scores, labels.gather(1, index), index

    def attach_prototype_bank(self, proto_bank: "VisualPrototypeBank"):
        self.prototype_bank = proto_bank
        return self

    def _detection_image_features_from_vision(self, vision_full, grid=None):
        feature_map = self.vision_model.post_layernorm(vision_full)
        class_token_out = torch.broadcast_to(feature_map[:, :1, :], feature_map[:, :-1].shape)

        feature_map = feature_map[:, 1:, :] * class_token_out
        feature_map = self.layer_norm(feature_map)

        rows, cols = grid or (self.sqrt_num_patches, self.sqrt_num_patches)
        feature_map = feature_map.reshape(feature_map.shape[0], rows, cols, feature_map.shape[-1])
        image_feats = feature_map.view(feature_map.shape[0], rows * cols, feature_map.shape[-1])
        return image_feats, feature_map

    def grid_box_bias(self, rows: int, cols: int) -> torch.Tensor:
        """``[rows * cols, 4]`` box bias of the top-left ``rows`` x ``cols`` patches."""
        P = self.sqrt_num_patches
        return self.box_bias.view(P, P, 4)[:rows, :cols].reshape(rows * cols, 4)

    def padding_free_grid(self, image_sizes) -> tuple[int, int]:
        """Patch rows and columns that hold image content, for ``(height, width)`` sizes.

        OWLv2 resizes the long side to ``image_size`` and pads the rest at the
        bottom and right; patches that are all padding can be dropped, which is
        free in accuracy (COCO zero-shot even gains ~0.008 mAP). A batch shares
        the largest grid: crop ``pixel_values[..., :rows * patch, :cols * patch]``.
        With ``set_preprocessing(long_side=...)`` the long side spans that many pixels.
        """
        P = (self.preprocess_long_side or self.image_size) // self.patch_size
        rows = cols = 1
        for height, width in image_sizes:
            side = max(height, width)
            rows = max(rows, math.ceil(height * P / side - 1e-6))
            cols = max(cols, math.ceil(width * P / side - 1e-6))
        return rows, cols

    def _image_grid_features(self, pixel_values):
        """
        Produces:
        image_feats: [B, P*P, vision_dim]  (same as your forward_object_detection pre-class_head)
        feature_map: [B, P, P, vision_dim]
        """
        _, vision_full = self._get_vision_outputs(pixel_values)
        return self._detection_image_features_from_vision(vision_full)

    def forward_proto_detection(
        self,
        pixel_values,
        prototype_bank: Optional["VisualPrototypeBank"] = None,
        box_delta: Optional[torch.Tensor] = None,
    ):
        """
        Returns:
        proto_logits:  [B, P*P, C*K] similarity to *prototypes*
        class_logits:  [B, P*P, C]   aggregated per class (logsumexp)
        objectness:    [B, P*P]
        pred_boxes:    [B, P*P, 4]   (cx,cy,w,h) in [0,1]
        extras:        (feature_map, class_embeds)
        """
        prototype_bank = prototype_bank if prototype_bank is not None else getattr(self, "prototype_bank", None)
        if prototype_bank is None:
            raise RuntimeError("Pass prototype_bank or call attach_prototype_bank() first.")
        B = pixel_values.shape[0]

        with torch.no_grad():
            image_feats, feature_map = self._image_grid_features(pixel_values)

        # [B, C*K, D_text] — these replace 'text_features'
        proto_embeds = prototype_bank.expand_for_batch(B)
        query_mask = None  # or torch.ones(B, proto_embeds.shape[1], dtype=torch.bool, device=image_feats.device)

        # Class logits over *prototypes*
        proto_logits, class_embeds = self.class_head(image_feats, proto_embeds, query_mask)
        # Aggregate to get per-class logits
        class_logits = prototype_bank.aggregate_logits(proto_logits, reduce="logsumexp")

        # Objectness & boxes unchanged
        objectness_logits = self.objectness_head(image_feats)[..., 0]
        pred_boxes = self.box_head(image_feats)
        if box_delta is not None:
            pred_boxes = pred_boxes + box_delta.view(1, 1, 4)
        pred_boxes = torch.sigmoid(pred_boxes + self.box_bias)

        return proto_logits, class_logits, objectness_logits, pred_boxes, (feature_map, class_embeds)


class PrototypeDetector(nn.Module):
    def __init__(
        self,
        owl: OwlV2,
        bank: VisualPrototypeBank,
        box_tuning_mode: str = "none",
        train_objectness_head: bool = False,
    ):
        super().__init__()
        if box_tuning_mode not in {"none", "bias", "last_layer", "full_head"}:
            raise ValueError(
                "box_tuning_mode must be one of: none, bias, last_layer, full_head"
            )
        self.owl = owl
        self.bank = bank
        self.box_tuning_mode = box_tuning_mode
        self.box_delta = nn.Parameter(torch.zeros(4)) if box_tuning_mode == "bias" else None

        for p in self.owl.parameters():
            p.requires_grad_(False)
        for p in self.bank.parameters():
            p.requires_grad_(True)
        if box_tuning_mode == "last_layer":
            for p in self.owl.box_head.dense2.parameters():
                p.requires_grad_(True)
        elif box_tuning_mode == "full_head":
            for p in self.owl.box_head.parameters():
                p.requires_grad_(True)
        if train_objectness_head:
            for p in self.owl.objectness_head.parameters():
                p.requires_grad_(True)

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def forward(self, pixel_values):
        return self.owl.forward_proto_detection(pixel_values, self.bank, self.box_delta)


    
if __name__ == '__main__':
    model = OwlV2()
    model = model.eval()
    model.cuda()
    with torch.no_grad():
        x = torch.randn(1, 3, 960, 960, device="cuda")
        token_id = torch.randint(0, 49408, (1, 16), device="cuda")
        attn_mask = torch.ones(1, 16, device="cuda")
        pred_logits, objectness_logits, pred_boxes, class_embeds, _ = (
            model.forward_object_detection(x, token_id, attn_mask)
        )
        print("pred_logits", pred_logits.shape)
        print("objectness_logits", objectness_logits.shape)
        print("pred_boxes", pred_boxes.shape)
