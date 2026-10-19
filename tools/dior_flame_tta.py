"""Training-free DIOR: rotation TTA and rotated FLAME support, base or large, TensorRT tower.

One pass over DIOR test scores every combination at once, so they all see the
same proposals:

- scorers: zero-shot, and FLAME for every ``--seeds`` support draw x
  ``--support-views`` (``none`` = the support images as they are, ``rot`` = each
  also rotated by 90/180/270 degrees with its boxes) x ``--alphas``;
- eval view sets: ``identity``, ``rot`` (4 views) and ``d4`` (8), nested, so the
  pass runs the largest ``--tta`` set and the smaller ones reuse its views.

Each view's detections go through the same per-patch argmax, NMS and top-100 as
``test_dior_flame.py``, then views are fused per (image, class) by WBF
``model_avg`` at IoU 0.55 (the evals harness's TTA rule: a box scores the mean of
every view's best match, a view that missed it counting 0) and capped at 100 per
image again. ``identity`` with no support views is exactly ``test_dior_flame.py``.

    uv run --extra train python tools/dior_flame_tta.py --model-size base --tensorrt \
        --save-metrics logs/rs_frozen/base.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from OWLv2torch import OwlV2
from OWLv2torch.torch_version.flame import Flame, select_detections
from ovd_eval import class_name_to_query, coco_eval_with_custom_sizes
from test_dior_flame import DATASETS, coco_gt, gt_boxes, per_class_ap, report_per_class

# Each (x, y) of a w x h image goes to (a x + b y + c, d x + e y + f) in the view
# (as openvocab_evals.tta).
TRANSPOSES = {
    "hflip": (Image.Transpose.FLIP_LEFT_RIGHT, lambda w, h: [[-1, 0, w], [0, 1, 0]]),
    "vflip": (Image.Transpose.FLIP_TOP_BOTTOM, lambda w, h: [[1, 0, 0], [0, -1, h]]),
    "rot90": (Image.Transpose.ROTATE_90, lambda w, h: [[0, 1, 0], [-1, 0, w]]),
    "rot180": (Image.Transpose.ROTATE_180, lambda w, h: [[-1, 0, w], [0, -1, h]]),
    "rot270": (Image.Transpose.ROTATE_270, lambda w, h: [[0, -1, h], [1, 0, 0]]),
    "transpose": (Image.Transpose.TRANSPOSE, lambda w, h: [[0, 1, 0], [1, 0, 0]]),
    "transverse": (Image.Transpose.TRANSVERSE, lambda w, h: [[0, -1, h], [-1, 0, w]]),
}
VIEW_SETS = {
    "identity": ("identity",),
    "rot": ("identity", "rot90", "rot180", "rot270"),
    "d4": ("identity", "rot90", "rot180", "rot270", "hflip", "vflip", "transpose", "transverse"),
}
WBF_IOU = 0.55
TOP_K = 100


def view_image(view: str, image: Image.Image) -> Image.Image:
    return image if view == "identity" else image.transpose(TRANSPOSES[view][0])


def map_boxes(view: str, boxes: torch.Tensor, width: int, height: int, inverse: bool) -> torch.Tensor:
    """xyxy boxes of the ``width x height`` source into ``view`` (or back, with ``inverse``)."""
    if view == "identity" or len(boxes) == 0:
        return boxes
    matrix = torch.eye(3, dtype=torch.float64)
    matrix[:2] = torch.tensor(TRANSPOSES[view][1](width, height), dtype=torch.float64)
    if inverse:
        matrix = torch.linalg.inv(matrix)
    matrix = matrix.to(device=boxes.device, dtype=boxes.dtype)
    linear, offset = matrix[:2, :2], matrix[:2, 2]
    a = boxes[:, :2] @ linear.T + offset
    b = boxes[:, 2:] @ linear.T + offset
    return torch.cat([torch.minimum(a, b), torch.maximum(a, b)], dim=1)


def svm_margins(flame: Flame, embeds: torch.Tensor) -> torch.Tensor:
    """``[P, C]`` RBF-SVM margins, NaN for queries without an SVM (as ``Flame.scores``)."""
    out = torch.full((len(embeds), len(flame.svms)), float("nan"), device=embeds.device)
    for c, svm in enumerate(flame.svms):
        if svm is None:
            continue
        x = (embeds - svm["mean"].to(embeds.device)) / svm["std"].to(embeds.device)
        sv = svm["vectors"].to(embeds.device)
        dist = (x * x).sum(1, keepdim=True) + (sv * sv).sum(1) - 2 * x @ sv.T
        out[:, c] = torch.exp(-svm["gamma"] * dist.clamp_min(0)) @ svm["coef"].to(embeds.device) + svm["bias"]
    return out


def build_model(args) -> OwlV2:
    if not args.tensorrt:
        return OwlV2(args.model_size).eval().to(args.device)
    from pseudo_label import build_trt_teacher

    args.trt_engine, args.checkpoint = None, None
    return build_trt_teacher(args).eval()


def support_draw(train, class_ids: list[int], to_query: dict[int, int], box_format: str, n: int, seed: int):
    """``test_dior_flame.py``'s support set: ``n`` train images per class, pooled."""
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {c: [] for c in class_ids}
    for i, objects in enumerate(train["objects"]):
        for c in set(objects["category"]) & by_class.keys():
            by_class[c].append(i)
    chosen = {c: rng.sample(by_class[c], min(n, len(by_class[c]))) for c in class_ids}
    support = sorted({i for c in class_ids for i in chosen[c]})
    out = []
    for i in support:
        sample = train[i]
        boxes, cats = gt_boxes(sample, box_format)
        keep = torch.tensor([c in to_query for c in cats.tolist()], dtype=torch.bool)
        labels = torch.tensor([to_query[c] for c in cats[keep].tolist()], dtype=torch.long)
        out.append((sample["image"].convert("RGB"), boxes[keep], labels))
    return out


# ---- WBF model_avg over numpy rows [image_id, label, x1, y1, x2, y2, score, view] ----

def wbf_group(boxes: np.ndarray, scores: np.ndarray, views: np.ndarray, n_views: int):
    """``openvocab_evals.fusion.weighted_boxes_fusion`` with ``conf_type='model_avg'``, equal weights."""
    n = len(scores)
    fused = np.zeros((n, 4))
    weighted_sum = np.zeros((n, 4))
    weight = np.zeros(n)
    view_best = np.zeros((n, n_views))
    k = 0
    for i in np.argsort(-scores, kind="stable"):
        box, v = boxes[i], float(scores[i])
        j = k
        if k:
            cand = fused[:k]
            iw = np.minimum(cand[:, 2], box[2]) - np.maximum(cand[:, 0], box[0])
            ih = np.minimum(cand[:, 3], box[3]) - np.maximum(cand[:, 1], box[1])
            inter = np.clip(iw, 0.0, None) * np.clip(ih, 0.0, None)
            union = (cand[:, 2] - cand[:, 0]) * (cand[:, 3] - cand[:, 1]) + (box[2] - box[0]) * (box[3] - box[1]) - inter
            iou = inter / np.maximum(union, 1e-12)
            m = int(iou.argmax())
            if iou[m] > WBF_IOU:
                j = m
        if j == k:
            k += 1
        weighted_sum[j] += v * box
        weight[j] += v
        view_best[j, views[i]] = max(view_best[j, views[i]], v)
        fused[j] = weighted_sum[j] / weight[j] if weight[j] > 0 else box
    return fused[:k], view_best[:k].sum(1) / n_views


def fuse_rows(rows: np.ndarray, n_views: int) -> np.ndarray:
    """WBF the views' rows per (image, class), then keep each image's ``TOP_K`` best."""
    if n_views == 1:
        return rows
    out = []
    order = np.lexsort((rows[:, 1], rows[:, 0]))
    rows = rows[order]
    keys = rows[:, 0] * 1000 + rows[:, 1]
    bounds = np.flatnonzero(np.diff(keys)) + 1
    for group in np.split(rows, bounds):
        boxes, scores = wbf_group(group[:, 2:6], group[:, 6], group[:, 7].astype(np.int64), n_views)
        block = np.empty((len(scores), 8))
        block[:, 0], block[:, 1], block[:, 2:6], block[:, 6], block[:, 7] = group[0, 0], group[0, 1], boxes, scores, 0
        out.append(block)
    fused = np.concatenate(out)
    keep = []
    order = np.lexsort((-fused[:, 6], fused[:, 0]))
    fused = fused[order]
    starts = np.flatnonzero(np.r_[True, np.diff(fused[:, 0]) != 0])
    ends = np.r_[starts[1:], len(fused)]
    for s, e in zip(starts, ends):
        keep.append(np.arange(s, min(e, s + TOP_K)))
    return fused[np.concatenate(keep)]


_DUMP: dict = {}


def _fuse_job(job):
    scorer, view_set, views = job
    rows = np.concatenate([_DUMP[(scorer, v)] for v in views])
    return scorer, view_set, fuse_rows(rows, len(views))


def to_records(rows: np.ndarray) -> list[dict]:
    return [{"image_id": int(r[0]), "category_id": int(r[1]),
             "bbox": [float(r[2]), float(r[3]), float(r[4] - r[2]), float(r[5] - r[3])], "score": float(r[6])}
            for r in rows]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-size", default="base", choices=["base", "large"])
    p.add_argument("--tensorrt", action="store_true", help="Run the vision tower as a TensorRT engine.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="FLAME support draws.")
    p.add_argument("--support-images", type=int, default=5)
    p.add_argument("--support-views", nargs="+", default=["none", "rot"], choices=["none", "rot", "d4"])
    p.add_argument("--alphas", type=float, nargs="+", default=[8.0])
    p.add_argument("--tta", default="d4", choices=list(VIEW_SETS), help="Largest eval view set; smaller ones are nested.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--fuse-workers", type=int, default=8)
    p.add_argument("--save-metrics", default=None)
    p.add_argument("--save-dump", default=None, help="Write per-view detections (npz) for later re-fusion.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dataset_id, eval_split, box_format, template = DATASETS["dior"]
    train = load_dataset(dataset_id, split="train")
    test = load_dataset(dataset_id, split=eval_split)
    if args.limit is not None:
        test = test.select(range(min(args.limit, len(test))))
    names: list[str] = test.features["objects"]["category"].feature.names
    class_ids = list(range(len(names)))
    to_query = {c: c for c in class_ids}
    queries = [class_name_to_query(n, template) for n in names]

    model = build_model(args)
    print(f"OWLv2 {args.model_size}, tower: {'TensorRT' if args.tensorrt else 'torch'}, "
          f"GPU {torch.cuda.get_device_name(torch.device(args.device))}")
    ref = Flame(model, queries, batch_size=args.batch_size)

    flames: dict[str, Flame] = {}
    for seed in args.seeds:
        support = support_draw(train, class_ids, to_query, box_format, args.support_images, seed)
        for sv in args.support_views:
            views = VIEW_SETS["identity" if sv == "none" else sv]
            flame = Flame(model, queries, batch_size=args.batch_size)
            start = time.perf_counter()
            for image, boxes, labels in support:
                for view in views:
                    flame.add_example(view_image(view, image),
                                      map_boxes(view, boxes, image.width, image.height, inverse=False), labels)
            flame.fit()
            n_sv = sum(len(s["vectors"]) for s in flame.svms if s is not None)
            print(f"FLAME seed {seed} support {sv}: {len(support)} images x {len(views)} views, "
                  f"{n_sv} support vectors, {time.perf_counter() - start:.1f}s")
            flames[f"s{seed}_{sv}"] = flame

    scorers = ["zero-shot"] + [f"flame_{key}_a{alpha:g}" for key in flames for alpha in args.alphas]
    eval_views = VIEW_SETS[args.tta]
    chunks: dict[tuple[str, str], list[np.ndarray]] = {(s, v): [] for s in scorers for v in eval_views}

    loader = DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=lambda b: b)
    checked = False
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"eval x{len(eval_views)} views"):
            images = [s["image"].convert("RGB") for s in batch]
            for v_idx, view in enumerate(eval_views):
                proposals = list(ref.proposals([view_image(view, im) for im in images]))
                # One margin pass per FLAME over the whole batch's patches, then split per image.
                all_embeds = torch.cat([prop[0] for prop in proposals])
                margins = {key: svm_margins(flame, all_embeds).split([len(prop[0]) for prop in proposals])
                           for key, flame in flames.items()}
                for i, (s, im, (embeds, logits, objectness, boxes)) in enumerate(zip(batch, images, proposals)):
                    boxes = map_boxes(view, boxes, im.width, im.height, inverse=True)
                    obj = torch.sigmoid(objectness)[:, None]
                    base = torch.sigmoid(logits) * obj
                    scored = {"zero-shot": base}
                    for key, flame in flames.items():
                        margin = margins[key][i]
                        has = ~torch.isnan(margin[0])
                        for alpha in args.alphas:
                            refined = base.clone()
                            refined[:, has] = torch.sigmoid(logits[:, has] + alpha * margin[:, has]) * obj
                            scored[f"flame_{key}_a{alpha:g}"] = refined
                        if not checked and flame.alpha in args.alphas:
                            assert torch.allclose(scored[f"flame_{key}_a{flame.alpha:g}"],
                                                  flame.scores(embeds, logits, objectness), atol=1e-5)
                    checked = True
                    for name, scores in scored.items():
                        b, sc, lab = select_detections(boxes, scores, threshold=0.0)
                        rows = torch.cat([torch.full_like(sc[:, None], float(s["image_id"])), lab[:, None].float(),
                                          b, sc[:, None], torch.full_like(sc[:, None], float(v_idx))], 1)
                        chunks[(name, view)].append(rows.double().cpu().numpy())
    elapsed = time.perf_counter() - start
    print(f"eval pass: {len(test)} images x {len(eval_views)} views in {elapsed:.0f}s "
          f"({len(test) * len(eval_views) / elapsed:.1f} views/s)")

    global _DUMP
    _DUMP = {k: np.concatenate(v) if v else np.zeros((0, 8)) for k, v in chunks.items()}
    del chunks
    if args.save_dump:
        Path(args.save_dump).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_dump, **{f"{s}|{v}": a for (s, v), a in _DUMP.items()})

    sets = [vs for vs in VIEW_SETS if set(VIEW_SETS[vs]) <= set(eval_views)]
    jobs = [(s, vs, VIEW_SETS[vs]) for s in scorers for vs in sets]
    start = time.perf_counter()
    with get_context("fork").Pool(args.fuse_workers) as pool:
        fused = {(s, vs): rows for s, vs, rows in pool.imap_unordered(_fuse_job, jobs)}
    print(f"fused {len(jobs)} (scorer, view set) pairs in {time.perf_counter() - start:.0f}s")

    gt = coco_gt(test, box_format, to_query, names)
    results = {}
    for s in scorers:
        for vs in sets:
            records = to_records(fused[(s, vs)])
            overall = coco_eval_with_custom_sizes(gt, records, names, title=f"{args.model_size} {s} tta={vs}")
            breakdown = report_per_class("dior", names, {s: per_class_ap(gt, records)})[s]
            results[f"{s}|{vs}"] = {"overall": overall, **breakdown}

    # Summary: mean over support seeds for each (support views, alpha, view set).
    print(f"\n=== summary: DIOR test, OWLv2 {args.model_size}, mean over seeds {args.seeds} ===")
    print(f"{'scorer':<26s} {'tta':<9s} {'mAP':>7s} {'small':>7s} {'shared':>7s} {'DIOR-only':>9s} {'spread':>7s}")
    summary = {}
    variants = ["zero-shot"] + [f"{sv}_a{alpha:g}" for sv in args.support_views for alpha in args.alphas]
    for variant in variants:
        members = ["zero-shot"] if variant == "zero-shot" else [f"flame_s{seed}_{variant}" for seed in args.seeds]
        for vs in sets:
            rs = [results[f"{m}|{vs}"] for m in members]
            maps = [r["overall"]["map"] for r in rs]
            row = {
                "map": float(np.mean(maps)),
                "map_small": float(np.mean([r["overall"]["map_small"] for r in rs])),
                "shared": float(np.mean([r["groups"]["shared with DOTA"] for r in rs])),
                "dior_only": float(np.mean([r["groups"]["DIOR-only"] for r in rs])),
                "dior_only_per_seed": [r["groups"]["DIOR-only"] for r in rs],
                "map_per_seed": maps,
            }
            summary[f"{variant}|{vs}"] = row
            print(f"{variant:<26s} {vs:<9s} {row['map']:>7.4f} {row['map_small']:>7.4f} {row['shared']:>7.4f} "
                  f"{row['dior_only']:>9.4f} {max(maps) - min(maps):>7.4f}")

    if args.save_metrics:
        def finite(value):
            if isinstance(value, dict):
                return {k: finite(v) for k, v in value.items()}
            if isinstance(value, list):
                return [finite(v) for v in value]
            return None if isinstance(value, float) and math.isnan(value) else value

        payload = {"model_size": args.model_size, "tensorrt": args.tensorrt, "seeds": args.seeds,
                   "support_images": args.support_images, "support_views": args.support_views,
                   "alphas": args.alphas, "tta": args.tta, "limit": args.limit, "wbf_iou": WBF_IOU,
                   "eval_seconds": elapsed, "summary": summary, "results": results}
        path = Path(args.save_metrics)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(finite(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Wrote metrics to {path}")


if __name__ == "__main__":
    main()
