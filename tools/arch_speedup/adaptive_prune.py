"""Adaptive keep: tokens whose learned score clears a threshold, padded to the batch max."""
import json, sys, torch
from arch_probe import *
import prune_scorer_lib as psl

model, _ = load_model(sys.argv[1])
val = CachedVal(model, AITOD / "subsets/vehicle_val200.json", AITOD / "val/images", batch_size=8)
state = torch.load("scorer_p3_probes.pt")
for li in (4, 6):
    p = psl.Probe(hidden=0).cuda(); p.load_state_dict(state[f"{li}_0"]); p.eval()
    for thr in (-6.0, -5.0, -4.0, -3.0):
        cfg = Cfg(prune=dict(after=li, keep=0.02, thresh=thr, scorer=lambda h, l, c, p=p: p(h[:, 1:]),
                             name="probe-h0", registers=16))
        det = ProbeDetector(model, ["a satellite photo of vehicle"], cfg)
        with torch.no_grad():
            m = evaluate(det, val)
        need, kept = det.ctx["need"], det.ctx["kept"]
        rec = {"variant": cfg.describe(), **m, "need_mean": sum(need) / len(need), "need_max": max(need),
               "kept_mean_b8": sum(kept) / len(kept)}
        print(json.dumps(rec), flush=True)
