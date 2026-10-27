"""Mask-only controls and register keeping for the learned pruning scorer."""
import json, sys, torch
from arch_probe import *
import prune_scorer_lib as psl

ckpt = sys.argv[1]
model, _ = load_model(ckpt)
val = CachedVal(model, AITOD / "subsets/vehicle_val200.json", AITOD / "val/images", batch_size=8)
prompts = ["a satellite photo of vehicle"]
state = torch.load("scorer_p3_probes.pt")
out = []
for li in (4, 6):
    p = psl.Probe(hidden=0).cuda()
    p.load_state_dict(state[f"{li}_0"])
    p.eval()
    feats = {}

    def capture(h, l, ctx, p=p):
        return p(h[:, 1:])

    # mask-only: score with the probe at layer li, but run the full model (capture the stream at li)
    class Grab:
        pass

    for keep in (0.1, 0.25):
        cfgs = [
            Cfg(prune=dict(after=li, keep=keep, scorer=capture, name="probe-h0", registers=16)),
        ]
        for cfg in cfgs:
            with torch.no_grad():
                m = evaluate(ProbeDetector(model, prompts, cfg), val)
            rec = {"variant": cfg.describe(), **m}
            print(json.dumps(rec), flush=True)
            out.append(rec)
# mask-only with the GT oracle (how much of the oracle's AP is masking alone)
for keep in (0.1, 0.25, 0.5):
    cfg = Cfg(prune=dict(after=12, keep=keep, scorer=gt_scorer(1), name="gt-d1 MASK-ONLY"))
    with torch.no_grad():
        m = evaluate(ProbeDetector(model, prompts, cfg), val)
    rec = {"variant": cfg.describe(), **m}
    print(json.dumps(rec), flush=True)
