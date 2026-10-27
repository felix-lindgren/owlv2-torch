"""Late token pruning on COCO, zero-shot: scorer distilled from the full model's final scores."""
import json, math, sys, torch, torch.nn.functional as F
from torch import nn
from torchvision.datasets import CocoDetection
from arch_probe import *
from prune_scorer_lib import Probe

model, _ = load_model(None)
ANN = "/mnt/datasets/coco/annotations/instances_val2017.json"
mini = json.load(open("/mnt/datasets/coco/annotations/instances_val2017_mini500.json"))
mini_ids = {im["id"] for im in mini["images"]}
full = CocoDetection(root="/mnt/datasets/coco/val2017", annFile=ANN, transform=model.image_transform_fast)
train_idx = [i for i, iid in enumerate(full.ids) if iid not in mini_ids][:240]
names = [full.coco.cats[c]["name"] for c in sorted(full.coco.getCatIds())]
det = ProbeDetector(model, [f"a photo of {n}" for n in names], Cfg())
LAYERS = (4, 6)
feats = {k: [] for k in LAYERS}
targets = []
vm = model.vision_model
with torch.no_grad(), torch.autocast("cuda"):
    for s in range(0, len(train_idx), 8):
        x = torch.stack([full[i][0] for i in train_idx[s:s + 8]]).cuda()
        B = x.shape[0]
        pe = vm.patch_embedding(x).flatten(2).transpose(1, 2)
        h = torch.cat([vm.class_embedding.expand(B, 1, -1).to(pe.dtype), pe], 1) + vm.position_embedding(vm.position_ids)
        h = vm.pre_layernorm(h)
        for li, layer in enumerate(vm.encoder.layers):
            if li in feats:
                feats[li].append(h[:, 1:].half().cpu())
            h = run_layer(layer, h, li, Cfg(), 60)
        image_feats, _ = model._detection_image_features_from_vision(h)
        qe = det.qe.unsqueeze(0).expand(B, -1, -1).to(image_feats.dtype)
        logits, _ = model.class_head(image_feats, qe, det.qm.unsqueeze(0).expand(B, -1))
        obj = model.objectness_head(image_feats)[..., 0]
        targets.append((torch.sigmoid(obj.float()) * torch.sigmoid(logits.float().amax(-1))).cpu())
Y = torch.cat(targets)
print("frac tokens with final score > 0.05:", (Y > 0.05).float().mean().item(), " > 0.3:", (Y > 0.3).float().mean().item())
probes = {}
for li in LAYERS:
    X = torch.cat(feats[li]).reshape(-1, 768)
    y = Y.reshape(-1)
    keep = (y > 0.01) | (torch.rand_like(y) < 0.2)
    X, y = X[keep].cuda(), y[keep].cuda()
    p = Probe(hidden=0).cuda()
    opt = torch.optim.AdamW(p.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000)
    for _ in range(3000):
        idx = torch.randint(0, len(y), (8192,), device="cuda")
        loss = F.binary_cross_entropy_with_logits(p(X[idx]), y[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    probes[li] = p.eval()
del feats
val = CachedVal(model, "/mnt/datasets/coco/annotations/instances_val2017_mini500.json", "/mnt/datasets/coco/val2017", batch_size=8)
prompts = [f"a photo of {n}" for n in val.class_names]
for li in LAYERS:
    p = probes[li]
    sc = lambda h, l, c, p=p: p(h[:, 1:])
    for cfg in [Cfg(prune=dict(after=li, keep=k, scorer=sc, name="probe-h0", registers=16)) for k in (0.25, 0.5)] + \
               [Cfg(prune=dict(after=li, keep=0.05, thresh=t, scorer=sc, name="probe-h0", registers=16)) for t in (-5.0, -4.0)]:
        d = ProbeDetector(model, prompts, cfg)
        with torch.no_grad():
            m = evaluate(d, val, point=False, top_k=300)
        rec = {"variant": cfg.describe(), **m}
        if "need" in d.ctx:
            rec["need_mean"] = sum(d.ctx["need"]) / len(d.ctx["need"]); rec["kept_b8"] = sum(d.ctx["kept"]) / len(d.ctx["kept"])
        print(json.dumps(rec), flush=True)
