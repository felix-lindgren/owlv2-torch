"""Training-free architecture probes for the OWLv2 vision tower.

Re-implements VisionTower.forward with optional changes, then scores the
trainer's point AP (coco_eval, conf 0.001, top-k 1500, objectness scoring) on
a cached val set. Variants (all inference-only, no retraining):

  skip       set of layer indices to skip entirely
  window     {layer: window size in patches}; CLS attends globally, patches
             attend within their window (+ the CLS key unless cls_key=False)
  kvpool     {layer: f}; patch keys/values average-pooled f x f (with a log f^2
             bias so the CLS key keeps its share of the softmax)
  prune      dict(after=k, keep=fraction, scorer=fn, mode="drop"|"kv"):
             from layer k on only the top-scored patch tokens are updated;
             the rest keep their layer-k state and are masked out of the
             detections. mode="kv" lets them still serve as keys/values.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import os

# OWL_REPO points the harness at another checkout of this repo (default: this one)
REPO = Path(os.environ.get("OWL_REPO", "/home/fellin/repos/openvocab/owlv2-torch"))
sys.path.insert(0, str(REPO))

from OWLv2torch.torch_version.owlv2 import OwlV2  # noqa: E402
from OWLv2torch.torch_version.point_metrics import DEFAULT_POINT_SIGMA  # noqa: E402
from OWLv2torch.utils.tokenizer import tokenize  # noqa: E402

AITOD = Path("/mnt/datasets/sat/xview_aitod_v2")


@dataclass
class Cfg:
    skip: set = field(default_factory=set)
    window: dict = field(default_factory=dict)
    cls_key: bool = True
    kvpool: dict = field(default_factory=dict)
    prune: dict | None = None
    merge: dict | None = None  # dict(after=k, frac=f): merge the most uniform f of 2x2 blocks
    pad_drop: bool = False  # drop tokens that lie wholly in the square padding
    registers: int = 0  # windowed/pooled layers: the R highest-norm patch tokens stay global

    def describe(self):
        parts = []
        if self.skip:
            parts.append(f"skip{sorted(self.skip)}")
        if self.window:
            ws = sorted(set(self.window.values()))
            parts.append(f"win{ws}@{sorted(self.window)}{'' if self.cls_key else '-nocls'}")
        if self.registers:
            parts.append(f"reg{self.registers}")
        if self.kvpool:
            parts.append(f"kvpool{sorted(self.kvpool.items())}")
        if self.prune:
            p = self.prune
            parts.append(f"prune(after={p['after']},keep={p['keep']}{',thr' + str(p['thresh']) if p.get('thresh') is not None else ''},{p.get('mode','drop')},{p.get('name','?')}"
                         f"{',reg' + str(p['registers']) if p.get('registers') else ''})")
        if self.merge:
            parts.append(f"merge2x2(after={self.merge['after']},frac={self.merge['frac']}"
                         f"{'' if self.merge.get('prop', True) else ',noprop'})")
        if self.pad_drop:
            parts.append("pad_drop")
        return " ".join(parts) or "baseline"


def register_idx(h, R):
    """Sequence indices (CLS = 0) of the R highest-norm patch tokens of ``h``."""
    return h[:, 1:].norm(dim=-1).topk(R, dim=1).indices + 1


def _take(t, idx):  # t [B, H, N, dh], idx [B, R] -> [B, H, R, dh]
    return t.gather(2, idx[:, None, :, None].expand(-1, t.shape[1], -1, t.shape[3]))


def window_attention(q, k, v, G, ws, scale, cls_key=True, reg=None):
    B, H, N, dh = q.shape
    nw = G // ws
    assert nw * ws == G, (G, ws)
    o_cls = F.scaled_dot_product_attention(q[:, :, :1], k, v, scale=scale)

    def win(t):
        return (
            t.reshape(B, H, nw, ws, nw, ws, dh)
            .permute(0, 2, 4, 1, 3, 5, 6)
            .reshape(B * nw * nw, H, ws * ws, dh)
        )

    qw, kw, vw = win(q[:, :, 1:]), win(k[:, :, 1:]), win(v[:, :, 1:])
    if cls_key:
        kc = k[:, :, :1].unsqueeze(1).expand(B, nw * nw, H, 1, dh).reshape(B * nw * nw, H, 1, dh)
        vc = v[:, :, :1].unsqueeze(1).expand(B, nw * nw, H, 1, dh).reshape(B * nw * nw, H, 1, dh)
        kw = torch.cat([kc, kw], 2)
        vw = torch.cat([vc, vw], 2)
    if reg is not None:
        R = reg.shape[1]
        kr = _take(k, reg).unsqueeze(1).expand(B, nw * nw, H, R, dh).reshape(B * nw * nw, H, R, dh)
        vr = _take(v, reg).unsqueeze(1).expand(B, nw * nw, H, R, dh).reshape(B * nw * nw, H, R, dh)
        kw = torch.cat([kw, kr], 2)
        vw = torch.cat([vw, vr], 2)
    ow = F.scaled_dot_product_attention(qw, kw, vw, scale=scale)
    ow = ow.reshape(B, nw, nw, H, ws, ws, dh).permute(0, 3, 1, 4, 2, 5, 6).reshape(B, H, G * G, dh)
    out = torch.cat([o_cls, ow], 2)
    if reg is not None:  # registers keep global attention
        o_r = F.scaled_dot_product_attention(_take(q, reg), k, v, scale=scale)
        out = out.scatter(2, reg[:, None, :, None].expand(-1, H, -1, dh), o_r)
    return out


def pooled_kv_attention(q, k, v, G, f, scale, reg=None):
    B, H, N, dh = q.shape
    g = G // f

    def pool(t):
        return t.reshape(B, H, g, f, g, f, dh).mean(dim=(3, 5)).reshape(B, H, g * g, dh)

    kp = torch.cat([k[:, :, :1], pool(k[:, :, 1:])], 2)
    vp = torch.cat([v[:, :, :1], pool(v[:, :, 1:])], 2)
    bias = torch.zeros(1 + g * g, device=q.device, dtype=q.dtype)
    bias[1:] = math.log(f * f)
    if reg is not None:  # registers also as full-resolution keys
        kp = torch.cat([kp, _take(k, reg)], 2)
        vp = torch.cat([vp, _take(v, reg)], 2)
        bias = torch.cat([bias, torch.zeros(reg.shape[1], device=q.device, dtype=q.dtype)])
    return F.scaled_dot_product_attention(q, kp, vp, attn_mask=bias.view(1, 1, 1, -1), scale=scale)


def run_layer(layer, h, li, cfg: Cfg, G, extra=None, key_bias=None):
    """One pre-LN encoder block. ``extra`` = frozen pruned tokens used as K/V only."""
    a = layer.self_attn
    B, N, d = h.shape
    H, dh = a.num_heads, a.head_dim
    y = layer.layer_norm1(h)
    q = a.q_proj(y).view(B, N, H, dh).transpose(1, 2)
    k = a.k_proj(y).view(B, N, H, dh).transpose(1, 2)
    v = a.v_proj(y).view(B, N, H, dh).transpose(1, 2)
    reg = register_idx(h, cfg.registers) if cfg.registers and (li in cfg.window or li in cfg.kvpool) else None
    if li in cfg.window:
        o = window_attention(q, k, v, G, cfg.window[li], a.scale, cfg.cls_key, reg)
    elif li in cfg.kvpool:
        o = pooled_kv_attention(q, k, v, G, cfg.kvpool[li], a.scale, reg)
    else:
        mask = key_bias
        if extra is not None:
            ye = layer.layer_norm1(extra)
            M = extra.shape[1]
            ke = a.k_proj(ye).view(B, M, H, dh).transpose(1, 2)
            ve = a.v_proj(ye).view(B, M, H, dh).transpose(1, 2)
            k = torch.cat([k, ke], 2)
            v = torch.cat([v, ve], 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=a.scale)
    o = o.transpose(1, 2).reshape(B, N, d)
    h = h + a.out_proj(o)
    h = h + layer.mlp(layer.layer_norm2(h))
    return h


def tower_forward(vm, x, cfg: Cfg, ctx: dict):
    """Returns (hidden [B, 1+P, d], token keep mask [B, P] or None)."""
    B = x.shape[0]
    pe = vm.patch_embedding(x.to(vm.patch_embedding.weight.dtype)).flatten(2).transpose(1, 2)
    P, d = pe.shape[1], pe.shape[2]
    G = math.isqrt(P)
    cls = vm.class_embedding.to(pe.dtype).expand(B, 1, -1)
    h = torch.cat([cls, pe], 1) + vm.position_embedding(vm.position_ids)
    h = vm.pre_layernorm(h)

    keep_mask = None
    key_bias = None
    if cfg.pad_drop:
        # Every image in ctx["content"] is (rows, cols) of patches with any image content.
        # Batch must share the same content shape (caller groups by it).
        rows, cols = ctx["content"]
        grid = torch.zeros(G, G, dtype=torch.bool, device=x.device)
        grid[:rows, :cols] = True
        idx = torch.nonzero(grid.flatten()).squeeze(1)
        gather = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device), idx + 1])
        h = h[:, gather]
        keep_mask = grid.flatten().unsqueeze(0).expand(B, -1)
        pad_canvas = torch.zeros(B, 1 + P, d, device=x.device, dtype=h.dtype)
        scatter_back = gather

    layers = vm.encoder.layers
    pruned_state = None
    merge_info = None
    for li, layer in enumerate(layers):
        if cfg.merge is not None and li == cfg.merge["after"]:
            h, key_bias, merge_info = merge_blocks(h, G, cfg.merge["frac"])
            if not cfg.merge.get("prop", True):
                key_bias = None
        if cfg.prune is not None and li == cfg.prune["after"]:
            P_full, P = P, h.shape[1] - 1  # tokens present (content tokens under pad_drop)
            scores = cfg.prune["scorer"](h, li, ctx)  # [B, P]
            if cfg.prune.get("registers"):
                scores = scores.float().scatter_add(
                    1, register_idx(h, cfg.prune["registers"]) - 1,
                    torch.full((B, cfg.prune["registers"]), 1e4, device=h.device))
            # keep_of_grid: the budget is a fraction of the full grid, capped at the tokens present
            base = P_full if cfg.prune.get("keep_of_grid") else P
            m = max(1, min(P, int(round(cfg.prune["keep"] * base))))
            if cfg.prune.get("thresh") is not None:
                # adaptive: as many tokens as the neediest image in the batch has above the
                # threshold (registers included), at least ``keep`` of the grid
                need = (scores > cfg.prune["thresh"]).sum(1)
                ctx.setdefault("need", []).extend((need / P).tolist())
                m = max(m, int(need.max()))
                ctx.setdefault("kept", []).extend([m / P] * B)
            top = scores.topk(m, dim=1).indices.sort(dim=1).values
            keep_mask = torch.zeros(B, P, dtype=torch.bool, device=x.device)
            keep_mask.scatter_(1, top, True)
            gather = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=x.device), top + 1], 1)
            full_h = h
            if cfg.prune.get("mode", "drop") == "kv":
                rest = (~keep_mask).nonzero()[:, 1].view(B, P - m) + 1
                pruned_state = h.gather(1, rest[..., None].expand(-1, -1, d))
            h = h.gather(1, gather[..., None].expand(-1, -1, d))
            P = P_full
        if li in cfg.skip:
            continue
        h = run_layer(layer, h, li, cfg, G, extra=pruned_state, key_bias=key_bias)

    if merge_info is not None:
        h = unmerge_blocks(h, G, merge_info)
    if cfg.prune is not None and cfg.prune["after"] >= len(layers):  # mask-only control
        scores = cfg.prune["scorer"](h, len(layers), ctx)
        m = max(1, int(round(cfg.prune["keep"] * P)))
        top = scores.topk(m, dim=1).indices
        keep_mask = torch.zeros(B, P, dtype=torch.bool, device=x.device).scatter_(1, top, True)
        return h, keep_mask
    if cfg.prune is not None:
        h = full_h.scatter(1, gather[..., None].expand(-1, -1, d), h)
    if cfg.pad_drop:
        pad_canvas[:, scatter_back] = h
        h = pad_canvas
        if cfg.prune is not None:  # kept content tokens, back on the full grid
            grid_keep = torch.zeros(B, P, dtype=torch.bool, device=x.device)
            grid_keep[:, scatter_back[1:] - 1] = keep_mask
            keep_mask = grid_keep
    return h, keep_mask


def _blocks(t, G):
    """[B, G*G, d] patch tokens -> [B, (G/2)^2, 4, d] 2x2 blocks."""
    B, _, d = t.shape
    g = G // 2
    return t.view(B, g, 2, g, 2, d).permute(0, 1, 3, 2, 4, 5).reshape(B, g * g, 4, d)


def merge_blocks(h, G, frac):
    """Average the ``frac`` most uniform 2x2 blocks into one token each.

    Returns the shorter sequence [CLS, merged..., unmerged tokens...], an additive
    key bias (log of each token's size, for proportional attention) and the
    bookkeeping to undo it.
    """
    B, N, d = h.shape
    blocks = _blocks(h[:, 1:], G)  # [B, nb, 4, d]
    nb = blocks.shape[1]
    f = F.normalize(F.layer_norm(blocks.float(), (d,)), dim=-1)
    sim = (f @ f.transpose(-1, -2)).sum((-1, -2)) - 4  # sum of off-diagonal cosines
    nm = int(round(frac * nb))
    order = sim.argsort(dim=1, descending=True)
    merged_idx, kept_idx = order[:, :nm], order[:, nm:]
    merged = blocks.gather(1, merged_idx[..., None, None].expand(-1, -1, 4, d)).mean(2)
    kept = blocks.gather(1, kept_idx[..., None, None].expand(-1, -1, 4, d)).reshape(B, -1, d)
    out = torch.cat([h[:, :1], merged, kept], 1)
    bias = torch.zeros(B, out.shape[1], device=h.device, dtype=h.dtype)
    bias[:, 1:1 + nm] = math.log(4)
    return out, bias[:, None, None, :], (merged_idx, kept_idx, nm)


def unmerge_blocks(h, G, info):
    merged_idx, kept_idx, nm = info
    B, _, d = h.shape
    nb = (G // 2) ** 2
    blocks = torch.empty(B, nb, 4, d, device=h.device, dtype=h.dtype)
    blocks.scatter_(1, merged_idx[..., None, None].expand(-1, -1, 4, d), h[:, 1:1 + nm, None, :].expand(-1, -1, 4, -1))
    blocks.scatter_(1, kept_idx[..., None, None].expand(-1, -1, 4, d), h[:, 1 + nm:].reshape(B, -1, 4, d))
    g = G // 2
    patches = blocks.view(B, g, g, 2, 2, d).permute(0, 1, 3, 2, 4, 5).reshape(B, G * G, d)
    return torch.cat([h[:, :1], patches], 1)


class ProbeDetector(nn.Module):
    def __init__(self, owl: OwlV2, prompts: list[str], cfg: Cfg):
        super().__init__()
        self.owl = owl
        self.cfg = cfg
        self.ctx = {}
        dev = owl.box_bias.device
        tok = tokenize(prompts, context_length=16, truncate=True).to(dev)
        with torch.no_grad():
            self.qe, self.qm = owl.encode_detection_queries(tok, tok == 0)

    def forward(self, pixel_values):
        owl = self.owl
        h, keep = tower_forward(owl.vision_model, pixel_values, self.cfg, self.ctx)
        image_feats, _ = owl._detection_image_features_from_vision(h)
        image_feats = owl.point_slot_features(image_feats)
        B = pixel_values.shape[0]
        qe = self.qe.unsqueeze(0).expand(B, -1, -1).to(image_feats.dtype)
        qm = self.qm.unsqueeze(0).expand(B, -1)
        logits, class_embeds = owl.class_head(image_feats, qe, qm)
        obj = owl.objectness_head(image_feats)[..., 0]
        boxes = torch.sigmoid(owl.box_head(image_feats) + owl.detection_box_bias)
        if keep is not None:
            S = owl.point_slot_grid
            G = owl.sqrt_num_patches
            km = keep.view(B, G, G)
            if S > 1:
                km = km.repeat_interleave(S, 1).repeat_interleave(S, 2)
            obj = obj.masked_fill(~km.reshape(B, -1), -1e4)
        return logits, obj, boxes, class_embeds, None


def load_model(ckpt: str | None, device="cuda", merge=True, image_size=None):
    from prototype_train.train_text import apply_text_checkpoint, get_vision_lora_config, merge_vision_lora

    model = OwlV2("base")
    if image_size:
        model.set_image_size(image_size)
    meta = None
    if ckpt:
        meta = torch.load(ckpt, map_location="cpu", weights_only=False)
        apply_text_checkpoint(model, meta)
        if merge and get_vision_lora_config(model) is not None:
            merge_vision_lora(model)
    return model.eval().to(device), meta


class CachedVal:
    """Preprocessed val batches kept in pinned host memory; sets ctx targets per batch."""

    def __init__(self, model, ann, images, batch_size=16, limit=None, cat_ids=None):
        from torchvision.datasets import CocoDetection
        from prototype_train.train import coco_collate_fn

        ds = CocoDetection(annFile=str(ann), root=str(images), transform=model.image_transform_fast)
        cat_ids = cat_ids or sorted(ds.coco.getCatIds())
        self.class_names = [ds.coco.cats[c]["name"] for c in cat_ids]
        collate = partial(
            coco_collate_fn,
            id2size=ds.coco.imgs,
            square_pad=True,
            category_id_to_label={c: i for i, c in enumerate(cat_ids)},
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, collate_fn=collate, num_workers=6)
        self.batches = []
        for i, b in enumerate(loader):
            if limit is not None and i * batch_size >= limit:
                break
            b["images"] = b["images"].half().pin_memory()
            self.batches.append(b)
        self.detector = None

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for b in self.batches:
            if self.detector is not None:
                self.detector.ctx["targets"] = b["targets"]
            yield b


def evaluate(detector, val: CachedVal, quiet=True, top_k=1500, point=True):
    import contextlib
    import io

    from prototype_train.train import coco_eval

    val.detector = detector
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext():
        m = coco_eval(
            detector,
            val,
            "cuda",
            confidence_threshold=0.001,
            score_with_objectness=True,
            top_k=top_k,
            output_format="standard",
            log_to_mlflow=False,
            point_sigma=DEFAULT_POINT_SIGMA if point else None,
        )
    keys = ["map", "map_50", "mar_50", "mar_50_very_tiny", "mar_50_tiny", "mar_50_small"] if point else [
        "map", "map_50", "map_small", "map_medium", "map_large", "mar_100"]
    return {k: float(m[k]) for k in keys if k in m}


# ---------------------------------------------------------------- scorers

def gt_scorer(dilate=1, jitter=1e-3):
    """Oracle: tokens whose patch (dilated by ``dilate``) holds a GT centre score 1."""

    def score(h, li, ctx):
        B, N, d = h.shape
        G = math.isqrt(N - 1)
        out = torch.rand(B, G, G, device=h.device) * jitter
        for i, t in enumerate(ctx["targets"]):
            if len(t["boxes"]) == 0:
                continue
            c = (t["boxes"][:, :2].to(h.device) * G).long().clamp(0, G - 1)
            grid = torch.zeros(G, G, device=h.device)
            grid[c[:, 1], c[:, 0]] = 1.0
            if dilate > 0:
                grid = F.max_pool2d(grid[None, None], 2 * dilate + 1, 1, dilate)[0, 0]
            out[i] += grid
        return out.flatten(1)

    return score


def random_scorer():
    def score(h, li, ctx):
        return torch.rand(h.shape[0], h.shape[1] - 1, device=h.device)

    return score
