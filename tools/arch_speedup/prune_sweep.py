"""Re-score a pruning checkpoint with its own trained scorer: all tokens, fixed keeps, adaptive thresholds.

python prune_sweep.py <final.pth> <prune block> [keeps, e.g. 0.1,0.25,0.4] [thresholds, e.g. -4,-3,-2]
For --image-size checkpoints set IMAGE_SIZE and OWL_REPO (a checkout with the patch applied)."""
import json, os, sys, torch
from torch import nn
from arch_probe import *

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
scorer_state = {k.split("token_scorer.", 1)[1]: v for k, v in ck["model_state_dict"].items() if "token_scorer." in k}
ck["model_state_dict"] = {k: v for k, v in ck["model_state_dict"].items() if "token_scorer." not in k}
tmp = f"noscorer_{os.getpid()}.pth"
torch.save(ck, tmp)
model, _ = load_model(tmp, image_size=int(os.environ["IMAGE_SIZE"]) if os.environ.get("IMAGE_SIZE") else None)
os.remove(tmp)
scorer = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 192), nn.GELU(), nn.Linear(192, 1)).cuda().eval()
scorer.load_state_dict(scorer_state)
after = int(sys.argv[2]) if len(sys.argv) > 2 else 6
val = CachedVal(model, AITOD / "subsets/vehicle_val200.json", AITOD / "val/images", batch_size=8)
score = lambda h, l, c: scorer(h[:, 1:].float()).squeeze(-1)
cfgs = [Cfg()]
keeps = [float(k) for k in sys.argv[3].split(",")] if len(sys.argv) > 3 else (0.05, 0.1, 0.15, 0.25, 0.4)
thrs = [float(t) for t in sys.argv[4].split(",")] if len(sys.argv) > 4 else (-4.0, -3.0, -2.0)
for keep in keeps:
    cfgs.append(Cfg(prune=dict(after=after, keep=keep, scorer=score, name="trained", registers=16)))
for thr in thrs:
    cfgs.append(Cfg(prune=dict(after=after, keep=0.02, thresh=thr, scorer=score, name="trained", registers=16)))
for cfg in cfgs:
    det = ProbeDetector(model, ["a satellite photo of vehicle"], cfg)
    with torch.no_grad():
        m = evaluate(det, val)
    rec = {"variant": cfg.describe(), **m}
    if "need" in det.ctx:
        rec["need_mean"] = sum(det.ctx["need"]) / len(det.ctx["need"])
        rec["kept_b8"] = sum(det.ctx["kept"]) / len(det.ctx["kept"])
    print(json.dumps(rec), flush=True)
