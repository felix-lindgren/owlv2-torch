"""python probe_run.py <aitod|coco> <ckpt|none> <variant-set> <out.jsonl>"""
import json, sys, time, torch
from arch_probe import *

dataset, ckpt, vset, out = sys.argv[1:5]
ckpt = None if ckpt == "none" else ckpt
model, meta = load_model(ckpt)
if dataset == "aitod":
    val = CachedVal(model, AITOD / "subsets/vehicle_val200.json", AITOD / "val/images", batch_size=8)
    prompts = [f"a satellite photo of {n}" for n in val.class_names]
    point = True
else:
    val = CachedVal(model, "/mnt/datasets/coco/annotations/instances_val2017_mini500.json", "/mnt/datasets/coco/val2017", batch_size=8)
    prompts = [f"a photo of {n}" for n in val.class_names]
    point = False

def variants(name):
    V = []
    if name in ("base", "all"):
        V.append(Cfg())
    if name in ("skip", "all"):
        V += [Cfg(skip={i}) for i in range(12)]
    if name in ("window", "all"):
        allL = set(range(12))
        vitdet = allL - {2, 5, 8, 11}
        V += [
            Cfg(window={i: 15 for i in allL}),
            Cfg(window={i: 15 for i in vitdet}),
            Cfg(window={i: 30 for i in vitdet}),
            Cfg(window={i: 15 for i in range(2, 9)}),
            Cfg(window={i: 30 for i in range(2, 9)}),
            Cfg(window={i: 15 for i in range(2, 7)}),
            Cfg(window={i: 15 for i in vitdet}, cls_key=False),
        ]
    if name in ("kvpool", "all"):
        V += [
            Cfg(kvpool={i: 2 for i in range(12)}),
            Cfg(kvpool={i: 2 for i in range(6)}),
            Cfg(kvpool={i: 2 for i in range(6, 12)}),
            Cfg(kvpool={i: 2 for i in range(2, 10)}),
            Cfg(kvpool={i: 3 for i in range(2, 10)}),
            Cfg(kvpool={i: 4 for i in range(2, 10)}),
        ]
    if name in ("reg", "all2"):
        allL = set(range(12))
        vitdet = allL - {2, 5, 8, 11}
        for R in (8, 16, 32):
            V.append(Cfg(window={i: 15 for i in vitdet}, registers=R))
        V += [
            Cfg(window={i: 15 for i in allL}, registers=16),
            Cfg(window={i: 15 for i in range(2, 9)}, registers=16),
            Cfg(window={i: 30 for i in vitdet}, registers=16),
            Cfg(window={i: 15 for i in range(1, 11)}, registers=16),
            Cfg(kvpool={i: 2 for i in allL}, registers=16),
            Cfg(kvpool={i: 2 for i in range(2, 10)}, registers=16),
            Cfg(kvpool={i: 4 for i in range(2, 10)}, registers=16),
        ]
    if name in ("merge3",):
        V += [
            Cfg(merge=dict(after=2, frac=0.5, prop=False)),
            Cfg(merge=dict(after=2, frac=0.75, prop=False)),
            Cfg(merge=dict(after=1, frac=0.5)),
            Cfg(merge=dict(after=3, frac=0.5)),
            Cfg(merge=dict(after=3, frac=0.6)),
            Cfg(merge=dict(after=2, frac=0.6)),
        ]
    if name in ("layout",):
        V += [
            Cfg(window={i: 15 for i in range(3, 9)}, registers=16),
            Cfg(window={i: 15 for i in range(2, 10)}, registers=16),
            Cfg(window={i: 20 for i in range(2, 9)}, registers=16),
            Cfg(window={i: 15 for i in range(1, 9)}, registers=16),
            Cfg(window={i: 10 for i in range(3, 8)}, registers=16),
            Cfg(kvpool={10: 2, 11: 2}),
            Cfg(kvpool={10: 4, 11: 4}),
            Cfg(kvpool={9: 2, 10: 2, 11: 2}),
            Cfg(kvpool={10: 4, 11: 4}, window={i: 15 for i in range(2, 9)}, registers=16),
        ]
    if name in ("skip2", "all2"):
        for s in ({10, 11}, {9, 10, 11}, {8, 9, 10, 11}, {6, 8, 10}, {7, 9, 11}):
            V.append(Cfg(skip=s))
    if name in ("merge", "all2"):
        for after in (2, 4, 6):
            for frac in (0.5, 0.75):
                V.append(Cfg(merge=dict(after=after, frac=frac)))
        V.append(Cfg(merge=dict(after=6, frac=1.0)))
        V.append(Cfg(merge=dict(after=8, frac=1.0)))
    if name in ("prune", "all") and dataset == "aitod":
        for after in (0, 3, 6, 9):
            for keep in (0.1, 0.25, 0.5):
                V.append(Cfg(prune=dict(after=after, keep=keep, scorer=gt_scorer(1), name="gt-d1")))
        for after in (3, 6):
            for keep in (0.1, 0.25):
                V.append(Cfg(prune=dict(after=after, keep=keep, scorer=gt_scorer(1), name="gt-d1", mode="kv")))
        V.append(Cfg(prune=dict(after=6, keep=0.25, scorer=random_scorer(), name="random")))
        for after in (3, 6):
            V.append(Cfg(prune=dict(after=after, keep=0.25, scorer=gt_scorer(1), name="gt-d1", registers=16)))
            V.append(Cfg(prune=dict(after=after, keep=0.1, scorer=gt_scorer(1), name="gt-d1", registers=16)))
    return V

with open(out, "a") as f:
    for cfg in variants(vset):
        det = ProbeDetector(model, prompts, cfg)
        t = time.time()
        with torch.no_grad():
            m = evaluate(det, val, point=point, top_k=1500 if point else 300)
        rec = {"dataset": dataset, "ckpt": ckpt, "variant": cfg.describe(), **m, "secs": round(time.time() - t, 1)}
        print(json.dumps(rec), flush=True)
        f.write(json.dumps(rec) + "\n")
