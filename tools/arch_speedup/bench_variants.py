"""Time architecture variants of the OWLv2 base vision tower (+ P3 heads) on one GPU.

Inference: forward only, fp16 weights, torch.compile, fp16 GEMM accumulation.
Training: autocast fp16, blocks 6-11 trainable, gradient flows through 0-5 (as the
hybrid LoRA forces), heads trainable; fwd + bwd, no optimizer.

python bench_variants.py infer|train [B]
"""
import copy
import math
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn

from arch_probe import Cfg, run_layer, load_model, merge_blocks, unmerge_blocks

torch.backends.cuda.matmul.allow_fp16_accumulation = True
mode = sys.argv[1]
if mode == "train":  # as the trainer's --attention-bwd-fp16-accumulation
    from OWLv2torch.torch_version.flash_attention_fp16acc import install

    install()
B = int(sys.argv[2]) if len(sys.argv) > 2 else (1 if mode == "infer" else 8)
only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
dev = "cuda"

model, _ = load_model(None, device=dev)
model.enable_point_slots(2)
G0 = 60


def shrink(model, mlp_keep=None, head_keep=None):
    """Copy with narrower MLPs / fewer heads (random weights, timing only)."""
    m = copy.deepcopy(model)
    for layer in m.vision_model.encoder.layers:
        if mlp_keep:
            d = layer.mlp[0].in_features
            h = int(layer.mlp[0].out_features * mlp_keep)
            layer.mlp[0] = nn.Linear(d, h).to(dev)
            layer.mlp[2] = nn.Linear(h, d).to(dev)
        if head_keep:
            a = layer.self_attn
            nh = int(a.num_heads * head_keep)
            d = a.embed_dim
            for n in ("q_proj", "k_proj", "v_proj"):
                setattr(a, n, nn.Linear(d, nh * a.head_dim).to(dev))
            a.out_proj = nn.Linear(nh * a.head_dim, d).to(dev)
            a.num_heads = nh
            a.embed_dim_inner = nh * a.head_dim
    return m


def run_layer_any(layer, h, li, cfg, G, extra=None, key_bias=None):
    # run_layer assumes embed_dim == num_heads*head_dim; handle pruned heads here.
    a = layer.self_attn
    if getattr(a, "embed_dim_inner", None) is None:
        return run_layer(layer, h, li, cfg, G, extra=extra, key_bias=key_bias)
    B_, N, d = h.shape
    H, dh = a.num_heads, a.head_dim
    y = layer.layer_norm1(h)
    q = a.q_proj(y).view(B_, N, H, dh).transpose(1, 2)
    k = a.k_proj(y).view(B_, N, H, dh).transpose(1, 2)
    v = a.v_proj(y).view(B_, N, H, dh).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, scale=a.scale)
    o = o.transpose(1, 2).reshape(B_, N, H * dh)
    h = h + a.out_proj(o)
    return h + layer.mlp(layer.layer_norm2(h))


def make_forward(m, cfg: Cfg, G=G0, keep_tokens=None, prune_after=None, prune_keep=None, prune_mode="drop",
                 skip=(), heads_on_kept=True, merge_after=None, merge_frac=None, merge_prop=True):
    vm = m.vision_model
    rows, cols = G if isinstance(G, tuple) else (G, G)
    P = rows * cols
    pos = vm.position_embedding.weight
    if (rows, cols) != (G0, G0):  # resample the position grid (timing only)
        grid = pos[1:].T.reshape(1, -1, G0, G0)
        grid = F.interpolate(grid, size=(rows, cols), mode="bilinear").reshape(-1, P).T
        pos = torch.cat([pos[:1], grid], 0)
    probe = nn.Linear(vm.class_embedding.shape[0], 1).to(dev, dtype=vm.class_embedding.dtype)

    def fwd(x):
        Bx = x.shape[0]
        pe = vm.patch_embedding(x).flatten(2).transpose(1, 2)
        h = torch.cat([vm.class_embedding.expand(Bx, 1, -1).to(pe.dtype), pe], 1) + pos
        h = vm.pre_layernorm(h)
        n_tok = P
        if keep_tokens is not None:  # training-time token drop from the input on
            n_tok = int(P * keep_tokens)
            idx = torch.rand(Bx, P, device=x.device).topk(n_tok, 1).indices + 1
            idx = torch.cat([torch.zeros(Bx, 1, dtype=torch.long, device=x.device), idx], 1)
            h = h.gather(1, idx[..., None].expand(-1, -1, h.shape[-1]))
        full = gather = extra = key_bias = merge_info = None
        for li, layer in enumerate(vm.encoder.layers):
            if merge_after is not None and li == merge_after:
                h, key_bias, merge_info = merge_blocks(h, G, merge_frac)
                if not merge_prop:
                    key_bias = None
            if prune_after is not None and li == prune_after:
                s = probe(h[:, 1:]).squeeze(-1)
                m_ = int(P * prune_keep)
                order = s.argsort(1, descending=True) + 1
                top = order[:, :m_].sort(1).values
                gather = torch.cat([torch.zeros(Bx, 1, dtype=torch.long, device=x.device), top], 1)
                full = h
                if prune_mode == "kv":
                    rest = order[:, m_:]
                    extra = h.gather(1, rest[..., None].expand(-1, -1, h.shape[-1]))
                h = h.gather(1, gather[..., None].expand(-1, -1, h.shape[-1]))
            if li in skip:
                continue
            h = run_layer_any(layer, h, li, cfg, G, extra=extra, key_bias=key_bias)
        if merge_info is not None:
            h = unmerge_blocks(h, G, merge_info)
        # heads (P3 slots) on all tokens, or only on the kept ones
        if full is not None and not heads_on_kept:
            h = full.scatter(1, gather[..., None].expand(-1, -1, h.shape[-1]), h)
        fm = vm.post_layernorm(h)
        fm = m.layer_norm(fm[:, 1:] * fm[:, :1])
        slots = m.point_slot_head(fm).view(Bx, -1, 4, fm.shape[-1]).flatten(1, 2)
        logits = m.class_head.dense0(slots)
        obj = m.objectness_head(slots)
        box = m.box_head(slots)
        return logits, obj, box

    return fwd


def variants():
    V = {}
    V["baseline"] = dict(cfg=Cfg())
    allL = set(range(12))
    vitdet = allL - {2, 5, 8, 11}
    V["win15 x8 (ViTDet)"] = dict(cfg=Cfg(window={i: 15 for i in vitdet}))
    V["win15 x12"] = dict(cfg=Cfg(window={i: 15 for i in allL}))
    V["win30 x8"] = dict(cfg=Cfg(window={i: 30 for i in vitdet}))
    V["win20 x8"] = dict(cfg=Cfg(window={i: 20 for i in vitdet}))
    V["win15 x8 + reg16"] = dict(cfg=Cfg(window={i: 15 for i in vitdet}, registers=16))
    V["win15 x12 + reg16"] = dict(cfg=Cfg(window={i: 15 for i in allL}, registers=16))
    V["merge2x2@4 frac0.5"] = dict(cfg=Cfg(), merge_after=4, merge_frac=0.5)
    V["merge2x2@4 frac0.75"] = dict(cfg=Cfg(), merge_after=4, merge_frac=0.75)
    V["merge2x2@6 frac1.0"] = dict(cfg=Cfg(), merge_after=6, merge_frac=1.0)
    V["merge2x2@2 frac0.5"] = dict(cfg=Cfg(), merge_after=2, merge_frac=0.5)
    V["merge2x2@2 frac0.75"] = dict(cfg=Cfg(), merge_after=2, merge_frac=0.75)
    V["merge2x2@2 frac0.75 noprop"] = dict(cfg=Cfg(), merge_after=2, merge_frac=0.75, merge_prop=False)
    V["pad drop 16:9 (60x34)"] = dict(cfg=Cfg(), G=(34, 60))
    V["kvpool2 x12"] = dict(cfg=Cfg(kvpool={i: 2 for i in allL}))
    V["kvpool2 x8 (2-9)"] = dict(cfg=Cfg(kvpool={i: 2 for i in range(2, 10)}))
    for after in (3, 6):
        for keep in (0.1, 0.25, 0.5):
            V[f"prune@{after} keep{keep}"] = dict(cfg=Cfg(), prune_after=after, prune_keep=keep)
        V[f"prune@{after} keep0.25 kv"] = dict(cfg=Cfg(), prune_after=after, prune_keep=0.25, prune_mode="kv")
    V["prune@6 keep0.25, heads all"] = dict(cfg=Cfg(), prune_after=6, prune_keep=0.25, heads_on_kept=False)
    # fine-tuned arms: E6 window layout, E2 keep 0.4 and adaptive means (B=1 mean, B=8 batch max)
    V["E6 win15 l2-8 + reg16"] = dict(cfg=Cfg(window={i: 15 for i in range(2, 9)}, registers=16))
    for after, keep, tag in ((6, 0.4, "0.40"), (6, 0.114, "adapt B1 0.114"), (6, 0.288, "adapt B8 0.288"),
                             (4, 0.1, "0.10"), (4, 0.25, "0.25"), (4, 0.4, "0.40"), (3, 0.4, "0.40"),
                             (3, 0.230, "adapt-3 B1 0.230"), (3, 0.536, "adapt-3 B8 0.536"),
                             (3, 0.356, "adapt-4 B1 0.356"), (3, 0.739, "adapt-4 B8 0.739"),
                             (4, 0.206, "adapt-3 B1 0.206"), (4, 0.494, "adapt-3 B8 0.494")):
        V[f"prune@{after} k{tag}"] = dict(cfg=Cfg(), prune_after=after, prune_keep=keep)
    # combinations: padding drop (COCO-like 4:3) + pruning, 800 px + pruning
    V["combo pad 4:3 (45x60)"] = dict(cfg=Cfg(), G=(45, 60))
    # 1800 tokens after block 6 = keep 0.5 of the full grid, the budget that held COCO AP (0.4959)
    V["combo pad 4:3 + prune@6 1800 tok"] = dict(cfg=Cfg(), G=(45, 60), prune_after=6, prune_keep=1800 / 2700)
    V["combo res800"] = dict(cfg=Cfg(), G=50)
    V["combo res800 + prune@3 k0.25"] = dict(cfg=Cfg(), G=50, prune_after=3, prune_keep=0.25)
    V["combo res800 + prune@3 k0.33"] = dict(cfg=Cfg(), G=50, prune_after=3, prune_keep=0.33)
    V["combo res1120"] = dict(cfg=Cfg(), G=70)
    # E8 (1120 px, prune after block 3) adaptive thresholds: mean keep (B=1, packed) and batch max (B=8 padded)
    for thr, b1, b8 in ((-4, 0.295, 0.646), (-3, 0.183, 0.448), (-2, 0.107, 0.293)):
        V[f"E8 res1120 prune@3 adapt{thr} B1 {b1}"] = dict(cfg=Cfg(), G=70, prune_after=3, prune_keep=b1)
        V[f"E8 res1120 prune@3 adapt{thr} B8 {b8}"] = dict(cfg=Cfg(), G=70, prune_after=3, prune_keep=b8)
    for keep in (0.1, 0.15, 0.25, 0.35):
        V[f"combo res1120 + prune@3 k{keep:.2f}"] = dict(cfg=Cfg(), G=70, prune_after=3, prune_keep=keep)
    V["res 800 (50x50)"] = dict(cfg=Cfg(), G=50)
    V["res 640 (40x40)"] = dict(cfg=Cfg(), G=40)
    V["skip 2 layers"] = dict(cfg=Cfg(), skip=(4, 7))
    V["skip 3 layers"] = dict(cfg=Cfg(), skip=(3, 6, 9))
    V["mlp 50%"] = dict(cfg=Cfg(), shrink=dict(mlp_keep=0.5))
    V["heads 8/12"] = dict(cfg=Cfg(), shrink=dict(head_keep=8 / 12))
    if mode == "train":
        V["token drop 50% (train only)"] = dict(cfg=Cfg(), keep_tokens=0.5)
        V["token drop 75% (train only)"] = dict(cfg=Cfg(), keep_tokens=0.25)
    return V


def bench(fn, x, train, iters=30, warm=8):
    def step():
        if train:
            out = fn(x)
            loss = sum(o.float().square().mean() for o in out)
            loss.backward()
        else:
            with torch.no_grad():
                fn(x)

    for _ in range(warm):
        step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        step()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


results = {}
for name, spec in variants().items():
    if only and not any(o in name for o in only):
        continue
    spec = dict(spec)
    m = shrink(model, **spec.pop("shrink")) if "shrink" in spec else model
    G = spec.pop("G", G0)
    if mode == "infer":
        m = m.half()
    else:
        m = m.float()
        for p in m.parameters():
            p.requires_grad_(False)
        for layer in m.vision_model.encoder.layers[6:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        for mod in (m.point_slot_head, m.class_head, m.objectness_head, m.box_head, m.layer_norm):
            for p in mod.parameters():
                p.requires_grad_(True)
    fwd = make_forward(m, G=G, **spec)
    rows, cols = G if isinstance(G, tuple) else (G, G)
    x = torch.randn(B, 3, rows * 16, cols * 16, device=dev, dtype=torch.half if mode == "infer" else torch.float)
    if mode == "train":
        x.requires_grad_(True)  # stands in for the LoRA: backward runs through blocks 0-5

        def fn(x, f=fwd):
            with torch.autocast("cuda", dtype=torch.float16):
                return f(x)
    else:
        fn = fwd
    cfn = torch.compile(fn)
    t0 = time.time()
    try:
        ms = bench(cfn, x, mode == "train")
    except Exception as exc:  # noqa: BLE001
        print(f"{name}: failed {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        continue
    results[name] = ms
    base = results.get("baseline", ms)
    print(f"{name:32s} {ms:8.2f} ms   {base / ms:5.2f}x   (compile+bench {time.time() - t0:.0f}s)", flush=True)
    if m is not model:
        del m
    torch._dynamo.reset()
    torch.cuda.empty_cache()
