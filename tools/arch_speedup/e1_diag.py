import json, sys, torch
from arch_probe import *
model, _ = load_model(sys.argv[1])
val = CachedVal(model, AITOD / "subsets/vehicle_val200.json", AITOD / "val/images", batch_size=8)
for cfg in (Cfg(),
            Cfg(prune=dict(after=0, keep=0.5, scorer=gt_scorer(1), name="gt-d1 (train regime)")),
            Cfg(prune=dict(after=12, keep=0.5, scorer=gt_scorer(1), name="gt-d1 MASK-ONLY"))):
    with torch.no_grad():
        m = evaluate(ProbeDetector(model, ["a satellite photo of vehicle"], cfg), val)
    print(json.dumps({"variant": cfg.describe(), **m}), flush=True)
