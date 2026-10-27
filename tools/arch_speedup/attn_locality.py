"""Per-layer attention statistics of the vision tower on a few images.

mean_dist: mean query->key distance in patches (patch queries, patch keys, weighted by attention)
cls: attention mass on the CLS key; win15/win20/win30: mass inside the query's aligned window
(windows as in the windowed variant, incl. CLS key in the numerator), r2/r4: mass within Chebyshev radius.
"""
import math, sys, json, torch, torch.nn.functional as F
from arch_probe import *

ckpt = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "none" else None
ann, imgs = (AITOD / "subsets/vehicle_val200.json", AITOD / "val/images")
if len(sys.argv) > 2 and sys.argv[2] == "coco":
    ann, imgs = ("/mnt/datasets/coco/annotations/instances_val2017_mini500.json", "/mnt/datasets/coco/val2017")
model, _ = load_model(ckpt)
val = CachedVal(model, ann, imgs, batch_size=4, limit=4)
x = val.batches[0]["images"].cuda().float()
vm = model.vision_model
G = 60
yy, xx = torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij")
pos = torch.stack([yy.flatten(), xx.flatten()], 1).float().cuda()  # [P, 2]
dist = torch.cdist(pos, pos, p=2)  # [P, P]
cheb = (pos[:, None] - pos[None]).abs().amax(-1)
win = {w: ((pos[:, None] // w) == (pos[None] // w)).all(-1) for w in (15, 20, 30)}
rows = []
with torch.no_grad():
    pe = vm.patch_embedding(x).flatten(2).transpose(1, 2)
    h = torch.cat([vm.class_embedding.expand(x.shape[0], 1, -1), pe], 1) + vm.position_embedding(vm.position_ids)
    h = vm.pre_layernorm(h)
    for li, layer in enumerate(vm.encoder.layers):
        a = layer.self_attn
        y = layer.layer_norm1(h)
        B, N, d = y.shape
        q = a.q_proj(y).view(B, N, a.num_heads, a.head_dim).transpose(1, 2)
        k = a.k_proj(y).view(B, N, a.num_heads, a.head_dim).transpose(1, 2)
        stats = {k_: 0.0 for k_ in ["mean_dist", "cls", "win15", "win20", "win30", "r1", "r2", "r4", "self"]}
        per_head_dist = torch.zeros(a.num_heads)
        cnt = 0
        for b in range(B):
            for hd in range(a.num_heads):
                att = torch.softmax((q[b, hd, 1:] @ k[b, hd].T) * a.scale, -1)  # [P, N]
                cls = att[:, 0]
                ap = att[:, 1:]
                md = (ap * dist).sum(-1) / ap.sum(-1)
                stats["mean_dist"] += md.mean().item()
                per_head_dist[hd] += md.mean().item() / B
                stats["cls"] += cls.mean().item()
                for w, m in win.items():
                    stats[f"win{w}"] += ((ap * m).sum(-1) + cls).mean().item()
                for r in (1, 2, 4):
                    stats[f"r{r}"] += (ap * (cheb <= r)).sum(-1).mean().item()
                stats["self"] += ap.diagonal().mean().item()
                cnt += 1
        stats = {k_: v / cnt for k_, v in stats.items()}
        stats["head_dist_min"] = per_head_dist.min().item(); stats["head_dist_max"] = per_head_dist.max().item()
        rows.append(stats)
        h = run_layer(layer, h, li, Cfg(), G)
print("layer mean_dist [head min-max]   cls   self   r1    r2    r4   win15 win20 win30")
for li, s in enumerate(rows):
    print(f"{li:5d} {s['mean_dist']:6.1f} [{s['head_dist_min']:4.1f}-{s['head_dist_max']:4.1f}]  {s['cls']:.3f} {s['self']:.3f} {s['r1']:.3f} {s['r2']:.3f} {s['r4']:.3f} {s['win15']:.3f} {s['win20']:.3f} {s['win30']:.3f}")
json.dump(rows, open(f"attn_locality_{'zs' if ckpt is None else 'ckpt'}_{sys.argv[2] if len(sys.argv)>2 else 'aitod'}.json", "w"))
