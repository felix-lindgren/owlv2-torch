"""COCO mini500 zero-shot: do padding-token drop and late pruning compose?

Distils the block-6 scorer as coco_scorer.py does (240 val2017 images outside mini500),
then scores at batch 1 (each image keeps its own content rectangle):
base, pad_drop, prune@6 keep 0.5, and pad_drop + prune@6 keep 0.5 of the content tokens.
"""
import json, math, time, torch, torch.nn.functional as F
from torchvision.datasets import CocoDetection
from arch_probe import *
from prune_scorer_lib import Probe

torch.manual_seed(0)
model, _ = load_model(None)
ANN = "/mnt/datasets/coco/annotations/instances_val2017.json"
MINI = "/mnt/datasets/coco/annotations/instances_val2017_mini500.json"
mini_ids = {im["id"] for im in json.load(open(MINI))["images"]}
full = CocoDetection(root="/mnt/datasets/coco/val2017", annFile=ANN, transform=model.image_transform_fast)
train_idx = [i for i, iid in enumerate(full.ids) if iid not in mini_ids][:240]
names = [full.coco.cats[c]["name"] for c in sorted(full.coco.getCatIds())]
det = ProbeDetector(model, [f"a photo of {n}" for n in names], Cfg())
LI = 6
feats, targets = [], []
vm = model.vision_model
with torch.no_grad(), torch.autocast("cuda"):
    for s in range(0, len(train_idx), 8):
        x = torch.stack([full[i][0] for i in train_idx[s:s + 8]]).cuda()
        B = x.shape[0]
        pe = vm.patch_embedding(x).flatten(2).transpose(1, 2)
        h = torch.cat([vm.class_embedding.expand(B, 1, -1).to(pe.dtype), pe], 1) + vm.position_embedding(vm.position_ids)
        h = vm.pre_layernorm(h)
        for li, layer in enumerate(vm.encoder.layers):
            if li == LI:
                feats.append(h[:, 1:].half().cpu())
            h = run_layer(layer, h, li, Cfg(), 60)
        image_feats, _ = model._detection_image_features_from_vision(h)
        qe = det.qe.unsqueeze(0).expand(B, -1, -1).to(image_feats.dtype)
        logits, _ = model.class_head(image_feats, qe, det.qm.unsqueeze(0).expand(B, -1))
        obj = model.objectness_head(image_feats)[..., 0]
        targets.append((torch.sigmoid(obj.float()) * torch.sigmoid(logits.float().amax(-1))).cpu())
X = torch.cat(feats).reshape(-1, 768)
y = torch.cat(targets).reshape(-1)
keep = (y > 0.01) | (torch.rand_like(y) < 0.2)
X, y = X[keep].cuda(), y[keep].cuda()
probe = Probe(hidden=0).cuda()
opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000)
for _ in range(3000):
    idx = torch.randint(0, len(y), (8192,), device="cuda")
    loss = F.binary_cross_entropy_with_logits(probe(X[idx]), y[idx])
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
probe.eval()
del X, y, feats

val = CachedVal(model, MINI, "/mnt/datasets/coco/val2017", batch_size=1)
ds = CocoDetection(annFile=MINI, root="/mnt/datasets/coco/val2017")
shapes = []
for i in range(len(val.batches)):
    im = ds.coco.imgs[ds.ids[i]]
    s = max(im["width"], im["height"])
    shapes.append((math.ceil(im["height"] / s * 60), math.ceil(im["width"] / s * 60)))
print(f"mean fraction of tokens with image content: {sum(r * c for r, c in shapes) / (3600 * len(shapes)):.3f}")


class ShapeIter(CachedVal):
    def __iter__(self):
        for b, s in zip(self.batches, shapes):
            self.detector.ctx["targets"] = b["targets"]
            self.detector.ctx["content"] = s
            yield b


val.__class__ = ShapeIter
prompts = [f"a photo of {n}" for n in val.class_names]
score = lambda h, l, c: probe(h[:, 1:])
prune = dict(after=LI, keep=0.5, scorer=score, name="probe-h0", registers=16)
import sys
cfgs = [Cfg(), Cfg(pad_drop=True), Cfg(prune=dict(prune)), Cfg(pad_drop=True, prune=dict(prune))]
if "grid" in sys.argv[1:]:  # pruning budget as a fraction of the full grid (same late-block tokens as prune alone)
    cfgs = [Cfg(prune=dict(prune)), Cfg(pad_drop=True, prune=dict(prune, keep_of_grid=True, name="probe-h0-grid"))]
for cfg in cfgs:
    d = ProbeDetector(model, prompts, cfg)
    t = time.time()
    with torch.no_grad():
        m = evaluate(d, val, point=False, top_k=300)
    print(json.dumps({"variant": cfg.describe(), **m, "secs": round(time.time() - t, 1)}), flush=True)
