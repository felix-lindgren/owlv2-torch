"""COCO mini500 zero-shot box mAP with and without the all-padding tokens (batch 1)."""
import json, math, sys, time, torch
from arch_probe import *

model, _ = load_model(None)
val = CachedVal(model, "/mnt/datasets/coco/annotations/instances_val2017_mini500.json", "/mnt/datasets/coco/val2017", batch_size=1)
from torchvision.datasets import CocoDetection
ds = CocoDetection(annFile="/mnt/datasets/coco/annotations/instances_val2017_mini500.json", root="/mnt/datasets/coco/val2017")
shapes = []
for i in range(len(val.batches)):
    im = ds.coco.imgs[ds.ids[i]]
    s = max(im["width"], im["height"])
    shapes.append((math.ceil(im["height"] / s * 60), math.ceil(im["width"] / s * 60)))
frac = sum(r * c for r, c in shapes) / (3600 * len(shapes))
print(f"mean fraction of tokens with image content: {frac:.3f}")
prompts = [f"a photo of {n}" for n in val.class_names]


class ShapeIter(CachedVal):
    def __iter__(self):
        for b, s in zip(self.batches, shapes):
            self.detector.ctx["targets"] = b["targets"]
            self.detector.ctx["content"] = s
            yield b


val.__class__ = ShapeIter
for cfg in (Cfg(), Cfg(pad_drop=True)):
    det = ProbeDetector(model, prompts, cfg)
    t = time.time()
    with torch.no_grad():
        m = evaluate(det, val, point=False, top_k=300)
    print(json.dumps({"variant": cfg.describe(), **m, "secs": round(time.time() - t, 1)}), flush=True)
