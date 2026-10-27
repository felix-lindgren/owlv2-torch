"""Zero-shot and FLAME OWLv2 on point-labelled ship sets (Sentinel-2 / Landsat), TensorRT tower.

The ship sets of ``../transformer-det`` (CLS/IMT, Finnish, Skylight S2 and
Landsat) label vessels as points, and their objects are 3-5 px. This writes
point predictions (box centers) for every scorer from one pass, for
transformer-det's point metrics to score (``ship_point_score.py`` there, or the
scratch scorer of the experiment):

- tilings: ``--tilings 0 288 160``: 0 is the whole image; ``T`` cuts T x T tiles
  (``--tile-overlap``) that are each upsampled to the model input, so a ship
  covers more of a patch. Every tiling is its own scorer set.
- zero-shot: one scorer per ``--queries`` entry, ``sigmoid(logit) * sigmoid(objectness)``.
- FLAME on ``--flame-query``: support images from ``--support-ann`` (``--support-images``
  with vessels per draw, ``--support-bg`` without), cut with the same tiling as the eval.
  The boxes are 3 x 3 px pseudo boxes or hull+wake boxes, so FLAME's IoU >= 0.7
  rule cannot select positives; instead a proposal is positive when its box
  center is within ``--pos-radius`` px of a vessel point (plus each point's
  nearest proposal within ``--pos-max-radius``) and negative beyond
  ``--neg-radius`` px of every point (the 50 best-scored plus 50 random, as
  ``Flame.add_example``).

Per image and scorer the tiles' centers are merged by a distance NMS (squares
of ``--nms-px`` at IoU 0.1, ~0.8 * nms-px apart) and the ``--top-k`` best kept.

    uv run --extra train python tools/ship_point_eval.py --tensorrt \
        --ann /mnt/datasets/sat/s2_finnish_vessels_coco/annotations/test.json \
        --image-root /mnt/datasets/sat/s2_finnish_vessels_coco/images \
        --support-ann /mnt/datasets/sat/s2_finnish_vessels_coco/annotations/train.json \
        --support-root /mnt/datasets/sat/s2_finnish_vessels_coco/images \
        --out logs/ships/fin_test.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import nms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from OWLv2torch.torch_version.flame import Flame
from dior_flame_tta import build_model, svm_margins
from ovd_eval import generate_tiles


def load_coco(path: str):
    d = json.loads(Path(path).read_text())
    points = defaultdict(list)
    for a in d["annotations"]:
        if a.get("iscrowd", 0):
            continue
        if "point" in a:
            points[a["image_id"]].append(a["point"])
        else:
            x, y, w, h = a["bbox"]
            points[a["image_id"]].append([x + w / 2, y + h / 2])
    return d["images"], {k: np.asarray(v, dtype=np.float32).reshape(-1, 2) for k, v in points.items()}


class Images(Dataset):
    def __init__(self, images: list[dict], root: str):
        self.images, self.root = images, Path(root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        return img["id"], Image.open(self.root / img["file_name"]).convert("RGB")


def crops(image: Image.Image, tile: int, overlap: int):
    """``[(x0, y0, crop)]``; tile 0 is the whole image."""
    if tile <= 0:
        return [(0, 0, image)]
    return [(x0, y0, image.crop((x0, y0, x1, y1)))
            for x0, y0, x1, y1 in generate_tiles(image.width, image.height, tile, overlap)]


def point_nms(points: torch.Tensor, scores: torch.Tensor, side: float, top_k: int, pre: int = 4000):
    if len(scores) > pre:
        keep = scores.topk(pre).indices
        points, scores = points[keep], scores[keep]
    boxes = torch.cat([points - side / 2, points + side / 2], 1)
    keep = nms(boxes, scores, 0.1)[:top_k]
    return points[keep], scores[keep]


def add_point_example(flame: Flame, image, points: np.ndarray, args) -> tuple[int, int]:
    """FLAME support from vessel points (see the module docstring); returns (positives, negatives)."""
    embeds, logits, objectness, boxes = next(flame.proposals([image]))
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    scores = torch.sigmoid(logits[:, 0]) * torch.sigmoid(objectness)
    if len(points):
        gt = torch.as_tensor(points, device=centers.device)
        dist = torch.cdist(centers, gt)  # [P, G]
        near = dist.amin(1)
        pos = near <= args.pos_radius
        nearest = dist.argmin(0)
        pos[nearest[dist[nearest, torch.arange(len(gt))] <= args.pos_max_radius]] = True
        negatives = (near > args.neg_radius).nonzero()[:, 0]
    else:
        pos = torch.zeros(len(centers), dtype=torch.bool, device=centers.device)
        negatives = torch.arange(len(centers), device=centers.device)
    hard = negatives[scores[negatives].argsort(descending=True)[:args.hard_negatives]]
    rest = negatives[~torch.isin(negatives, hard)]
    easy = rest[torch.randperm(len(rest), generator=flame._rng)[:args.random_negatives].to(rest.device)]
    flame._append(0, embeds[pos], 1)
    flame._append(0, embeds[torch.cat([hard, easy])], 0)
    return int(pos.sum()), len(hard) + len(easy)


def fit_active_set(flame: Flame, *, start_frac: float = 0.25, max_rounds: int = 8, tol: float = 1e-3) -> Flame:
    """``Flame.fit`` for query 0 by working-set rounds; the same SVM, ~4x faster at 77k examples.

    Standardization, gamma (sklearn's "scale") and the class weights ("balanced")
    are fixed from all examples, so every round solves a sub-problem of the full
    QP. Start from all positives plus ``start_frac`` of the negatives; after each
    fit, add the examples outside the set that violate the margin (y f < 1 - tol)
    and keep only the support vectors of the set. No violator left is the full
    solution (checked: same support vectors, margins within 1e-3); a set that has
    not converged in ``max_rounds`` falls back to ``Flame.fit``.
    """
    from sklearn.svm import SVC

    x, y = (torch.cat(t) for t in zip(*flame.examples[0]))
    x, y = x.float(), y.numpy().astype(np.int64)
    if y.min() == y.max():
        return flame.fit()
    mean, std = x.mean(0), x.std(0) + 1e-6
    z = (x - mean) / std
    zn = z.numpy().astype(np.float64)
    zg = z.to(flame.device)
    gamma = 1.0 / (zn.shape[1] * zn.var())
    n, n_pos = len(y), int(y.sum())
    weights = {1: n / (2 * n_pos), 0: n / (2 * (n - n_pos))}
    signs = np.where(y == 1, 1.0, -1.0)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    idx = np.r_[pos, np.random.default_rng(0).choice(neg, int(len(neg) * start_frac), replace=False)]
    for _ in range(max_rounds):
        svc = SVC(C=flame.svm_c, kernel="rbf", gamma=gamma, class_weight=weights, cache_size=4000).fit(zn[idx], y[idx])
        sv = zg[torch.as_tensor(idx[svc.support_], device=zg.device)]
        coef = torch.as_tensor(svc.dual_coef_[0], dtype=torch.float32, device=zg.device)
        f = torch.cat([torch.exp(-gamma * torch.cdist(c, sv).square()) @ coef for c in zg.split(16384)])
        f = f.cpu().numpy() + svc.intercept_[0]
        violators = np.setdiff1d(np.flatnonzero(signs * f < 1 - tol), idx)
        if not len(violators):
            flame.svms[0] = {"mean": mean, "std": std, "gamma": gamma, "vectors": sv.cpu(),
                             "coef": coef.cpu(), "bias": float(svc.intercept_[0])}
            return flame
        idx = np.r_[idx[svc.support_], violators]
    print(f"[active set] not converged after {max_rounds} rounds; full fit")
    return flame.fit()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ann", required=True, help="COCO file of the eval images")
    p.add_argument("--image-root", required=True)
    p.add_argument("--subset", default="all", choices=["all", "ships"], help="ships: only images with vessels")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--model-size", default="base", choices=["base", "large"])
    p.add_argument("--tensorrt", action="store_true")
    p.add_argument("--queries", nargs="+", default=["ship", "boat", "a satellite image of a ship"])
    p.add_argument("--tilings", type=int, nargs="+", default=[0])
    p.add_argument("--tile-overlap", type=int, default=64)
    p.add_argument("--flame-query", default=None, help="one of --queries; FLAME runs only when set")
    p.add_argument("--support-ann", default=None)
    p.add_argument("--support-root", default=None)
    p.add_argument("--support-images", type=int, nargs="+", default=[5])
    p.add_argument("--support-bg", type=int, default=5, help="vessel-free support images per draw")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--alphas", type=float, nargs="+", default=[8.0])
    p.add_argument("--pos-radius", type=float, default=1.5)
    p.add_argument("--pos-max-radius", type=float, default=4.0)
    p.add_argument("--neg-radius", type=float, default=8.0)
    p.add_argument("--hard-negatives", type=int, default=50)
    p.add_argument("--random-negatives", type=int, default=50)
    p.add_argument("--nms-px", type=float, default=3.0)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--full-fit", action="store_true", help="Flame.fit instead of the (identical) active-set fit")
    p.add_argument("--flame-cache", default=None, help="pickle of fitted SVMs, reused when the support config matches")
    p.add_argument("--view-chunk", type=int, default=64, help="views per tower call / margin pass")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    images, points = load_coco(args.ann)
    if args.subset == "ships":
        images = [im for im in images if im["id"] in points]
    if args.limit:
        images = images[:args.limit]

    model = build_model(args)
    print(f"OWLv2 {args.model_size}, tower: {'TensorRT' if args.tensorrt else 'torch'}, "
          f"GPU {torch.cuda.get_device_name(torch.device(args.device))}; {len(images)} images, "
          f"{sum(len(points.get(im['id'], ())) for im in images)} vessels")
    ref = Flame(model, args.queries, batch_size=args.batch_size)

    # flames[tiling] = [(name, flame, alpha)]
    flames: dict[int, list[tuple[str, Flame, float]]] = {t: [] for t in args.tilings}
    support_log = []
    if args.flame_query:
        qi = args.queries.index(args.flame_query)
        s_images, s_points = load_coco(args.support_ann)
        ship_pool = [im for im in s_images if im["id"] in s_points]
        bg_pool = [im for im in s_images if im["id"] not in s_points]
        root = Path(args.support_root)
        cache_path = Path(args.flame_cache) if args.flame_cache else None
        cache = pickle.loads(cache_path.read_bytes()) if cache_path and cache_path.exists() else {}
        config = (args.model_size, args.flame_query, args.support_ann, args.support_bg, args.tile_overlap,
                  args.pos_radius, args.pos_max_radius, args.neg_radius, args.hard_negatives, args.random_negatives)
        for seed in args.seeds:
            for n in args.support_images:
                rng = random.Random(seed)
                chosen = rng.sample(ship_pool, min(n, len(ship_pool))) + rng.sample(bg_pool, min(args.support_bg, len(bg_pool)))
                loaded = None
                for t in args.tilings:
                    flame = Flame(model, [args.flame_query], batch_size=args.batch_size)
                    key = repr((config, seed, n, t))
                    if key in cache:
                        flame.svms[0], log = cache[key]
                    else:
                        if loaded is None:
                            loaded = [(Image.open(root / im["file_name"]).convert("RGB"),
                                       s_points.get(im["id"], np.zeros((0, 2), np.float32))) for im in chosen]
                        n_pos = n_neg = 0
                        for image, pts in loaded:
                            for x0, y0, crop in crops(image, t, args.tile_overlap):
                                local = pts - np.array([x0, y0], np.float32)
                                inside = (local[:, 0] >= 0) & (local[:, 1] >= 0) & (local[:, 0] < crop.width) & (local[:, 1] < crop.height)
                                a, b = add_point_example(flame, crop, local[inside], args)
                                n_pos, n_neg = n_pos + a, n_neg + b
                        fit_start = time.perf_counter()
                        flame.fit() if args.full_fit else fit_active_set(flame)
                        n_sv = sum(len(s["vectors"]) for s in flame.svms if s is not None)
                        log = {"tiling": t, "seed": seed, "n": n, "vessels": sum(len(pts) for _, pts in loaded),
                               "positives": n_pos, "negatives": n_neg, "support_vectors": n_sv,
                               "fit_seconds": time.perf_counter() - fit_start}
                        flame.examples = [[]]
                        if cache_path:
                            cache[key] = (flame.svms[0], log)
                            cache_path.parent.mkdir(parents=True, exist_ok=True)
                            cache_path.write_bytes(pickle.dumps(cache))
                    print(f"FLAME t{t} seed {seed} n{n}+{args.support_bg}bg: {log['vessels']} vessels -> "
                          f"{log['positives']} positives, {log['negatives']} negatives, "
                          f"{log['support_vectors']} support vectors, fit {log.get('fit_seconds', float('nan')):.0f}s")
                    support_log.append(log)
                    for alpha in args.alphas:
                        flames[t].append((f"flame_n{n}_s{seed}_a{alpha:g}", flame, alpha))

    preds: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
    loader = DataLoader(Images(images, args.image_root), batch_size=args.batch_size,
                        num_workers=args.num_workers, collate_fn=lambda b: b)
    start = time.perf_counter()
    n_views = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="eval"):
            for t in args.tilings:
                views = [(image_id, image.size, x0, y0, crop) for image_id, image in batch
                         for x0, y0, crop in crops(image, t, args.tile_overlap)]
                n_views += len(views)
                per_image: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
                # Chunks of views: 8 windows x 36 tiles of embeddings do not fit at once.
                for c0 in range(0, len(views), args.view_chunk):
                    chunk = views[c0:c0 + args.view_chunk]
                    proposals = list(ref.proposals([v[4] for v in chunk]))
                    all_embeds = torch.cat([prop[0] for prop in proposals])
                    margins = {name: torch.cat([svm_margins(flame, e)[:, 0] for e in all_embeds.split(16384)])
                               .split([len(prop[0]) for prop in proposals])
                               for name, flame, _ in flames[t]}
                    del all_embeds
                    for k, ((image_id, size, x0, y0, _), (_, logits, objectness, boxes)) in enumerate(zip(chunk, proposals)):
                        centers = (boxes[:, :2] + boxes[:, 2:]) / 2 + boxes.new_tensor([x0, y0])
                        obj = torch.sigmoid(objectness)
                        ok = (centers[:, 0] >= 0) & (centers[:, 1] >= 0) & (centers[:, 0] < size[0]) & (centers[:, 1] < size[1])
                        scored = {f"zs_{q}": torch.sigmoid(logits[:, i]) * obj for i, q in enumerate(args.queries)}
                        for name, flame, alpha in flames[t]:
                            scored[name] = torch.sigmoid(logits[:, qi] + alpha * margins[name][k]) * obj
                        for name, s in scored.items():
                            per_image[image_id][name].append((centers[ok], s[ok]))
                    del proposals, margins
                for image_id, by_scorer in per_image.items():
                    for name, parts in by_scorer.items():
                        pts, sc = point_nms(torch.cat([a for a, _ in parts]), torch.cat([b for _, b in parts]),
                                            args.nms_px, args.top_k)
                        preds[f"{name}@t{t}"][image_id] = (pts.float().cpu().numpy(), sc.float().cpu().numpy())
    elapsed = time.perf_counter() - start
    print(f"eval pass: {len(images)} images, {n_views} views in {elapsed:.0f}s ({n_views / elapsed:.1f} views/s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"args": vars(args), "support": support_log, "eval_seconds": elapsed,
                     "image_ids": [im["id"] for im in images], "preds": dict(preds)}, f)
    print(f"wrote {len(preds)} scorers to {out}")


if __name__ == "__main__":
    main()
