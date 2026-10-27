"""Learn a cheap token scorer at block k that predicts the full model's final per-token score.

1. Run the trained model on train images; record the residual stream entering blocks
   in LAYERS and the final per-token max-slot score sigmoid(obj) * sigmoid(class).
2. Fit a probe per layer (standardise -> linear, and standardise -> 768x128 MLP) with BCE
   against the soft final score (pure distillation, no labels).
3. On val200: recall of the full model's confident tokens inside the probe's top-rho,
   then point AP with the probe as the pruning scorer.

python prune_scorer.py <ckpt> <out-prefix>
"""
import json
import math
import random
import sys

import torch
import torch.nn.functional as F
from torch import nn

from arch_probe import AITOD, Cfg, CachedVal, ProbeDetector, evaluate, load_model, run_layer, gt_scorer

ckpt, out = sys.argv[1], sys.argv[2]
LAYERS = [3, 4, 6]
N_TRAIN = 240
torch.manual_seed(0)
model, meta = load_model(ckpt)
prompts = ["a satellite photo of vehicle"]
det_full = ProbeDetector(model, prompts, Cfg())


def collect(batches):
    feats = {k: [] for k in LAYERS}
    targets = []
    vm = model.vision_model
    with torch.no_grad(), torch.autocast("cuda"):
        for b in batches:
            x = b["images"].cuda()
            B = x.shape[0]
            pe = vm.patch_embedding(x).flatten(2).transpose(1, 2)
            h = torch.cat([vm.class_embedding.expand(B, 1, -1).to(pe.dtype), pe], 1) + vm.position_embedding(vm.position_ids)
            h = vm.pre_layernorm(h)
            for li, layer in enumerate(vm.encoder.layers):
                if li in feats:
                    feats[li].append(h[:, 1:].half().cpu())
                h = run_layer(layer, h, li, Cfg(), 60)
            logits, obj, _, _, _ = det_full.forward_from_hidden(h)
            score = torch.sigmoid(obj.float()) * torch.sigmoid(logits.float().amax(-1))
            S = model.point_slot_grid
            score = score.view(B, 60, S, 60, S).amax(dim=(2, 4)).flatten(1)
            targets.append(score.cpu())
    return {k: torch.cat(v) for k, v in feats.items()}, torch.cat(targets)


def forward_from_hidden(self, h):
    owl = self.owl
    image_feats, _ = owl._detection_image_features_from_vision(h)
    image_feats = owl.point_slot_features(image_feats)
    B = h.shape[0]
    qe = self.qe.unsqueeze(0).expand(B, -1, -1).to(image_feats.dtype)
    qm = self.qm.unsqueeze(0).expand(B, -1)
    logits, ce = owl.class_head(image_feats, qe, qm)
    obj = owl.objectness_head(image_feats)[..., 0]
    boxes = torch.sigmoid(owl.box_head(image_feats) + owl.detection_box_bias)
    return logits, obj, boxes, ce, None


ProbeDetector.forward_from_hidden = forward_from_hidden

# train images: a random subset of train_all (no overlap with val200)
train = CachedVal(model, AITOD / "subsets/vehicle_train_all.json", AITOD / "train/images", batch_size=8, limit=N_TRAIN)
val = CachedVal(model, AITOD / "subsets/vehicle_val200.json", AITOD / "val/images", batch_size=8)
Xtr, Ytr = collect(train.batches)
Xva, Yva = collect(val.batches)
print("train tokens", Ytr.numel(), "frac>0.05", (Ytr > 0.05).float().mean().item(), "frac>0.3", (Ytr > 0.3).float().mean().item())


class Probe(nn.Module):
    def __init__(self, d=768, hidden=0):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.net = nn.Linear(d, 1) if not hidden else nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(self.norm(x.float())).squeeze(-1)


def fit(X, Y, hidden):
    X = X.reshape(-1, X.shape[-1])
    Y = Y.reshape(-1)
    # all confident tokens + a random 20% of the rest keeps the fit fast
    keep = (Y > 0.01) | (torch.rand_like(Y) < 0.2)
    X, Y = X[keep].cuda(), Y[keep].cuda()
    p = Probe(hidden=hidden).cuda()
    opt = torch.optim.AdamW(p.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(Y)
    steps = 3000
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        idx = torch.randint(0, n, (8192,), device="cuda")
        loss = F.binary_cross_entropy_with_logits(p(X[idx]), Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return p.eval()


def recall_at(scores, Y, keep, thr):
    """Fraction of the full model's tokens with score > thr that land in the top-keep."""
    m = int(round(keep * scores.shape[1]))
    top = scores.topk(m, 1).indices
    kept = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, top, True)
    pos = Y > thr
    return (kept & pos).sum().item() / max(1, pos.sum().item())


results = []
probes = {}
for li in LAYERS:
    for hidden in (0, 128):
        p = fit(Xtr[li], Ytr, hidden)
        with torch.no_grad():
            sva = torch.cat([p(Xva[li][i:i + 8].cuda()).cpu() for i in range(0, len(Xva[li]), 8)])
        row = {"layer": li, "hidden": hidden}
        for keep in (0.1, 0.25, 0.5):
            for thr in (0.05, 0.3):
                row[f"rec@{keep}>{thr}"] = round(recall_at(sva, Yva, keep, thr), 4)
        print(json.dumps(row), flush=True)
        results.append(row)
        probes[(li, hidden)] = p
torch.save({f"{k[0]}_{k[1]}": v.state_dict() for k, v in probes.items()}, f"{out}_probes.pt")


def probe_scorer(p):
    def score(h, li, ctx):
        return p(h[:, 1:])

    return score


# point AP with the learned scorer (drop and kv modes)
for li in (3, 4, 6):
    for hidden in (0, 128):
        for keep in (0.1, 0.25, 0.5):
            for mode in ("drop", "kv"):
                if mode == "kv" and keep == 0.5:
                    continue
                cfg = Cfg(prune=dict(after=li, keep=keep, scorer=probe_scorer(probes[(li, hidden)]), mode=mode,
                                     name=f"probe-h{hidden}"))
                with torch.no_grad():
                    m = evaluate(ProbeDetector(model, prompts, cfg), val)
                rec = {"variant": cfg.describe(), **m}
                print(json.dumps(rec), flush=True)
                results.append(rec)
json.dump(results, open(f"{out}_results.json", "w"), indent=1)
