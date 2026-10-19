"""Diagnose why OWLv2 misses small objects on DIOR.

Two probes:

1. Patch-collision ceiling (no model): map every GT box centre onto the
   ViT token grid and count how many GTs share a patch with another GT.
   Each token emits one box, so with local predictions only one GT per
   occupied patch is recoverable. Reported at native zoom and at the
   tile zoom used in probe 2.

2. Zoom test: run the same weights on the full image and on overlapping
   tiles (each upsampled to the model input), then score full / tiles /
   full+tiles per size bucket. Detections are cached so re-scoring is cheap.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ovd_eval import (
    AREA_RANGES,
    DiorDetectionDataset,
    build_owlv2_model,
    generate_tiles,
    merge_tiled_detections,
    run_owlv2_inference,
)


class TiledDataset(Dataset):
    def __init__(self, base: DiorDetectionDataset, tile_size: int, overlap: int):
        self.base = base
        self.items = []
        for idx in range(len(base)):
            w, h = int(base.ds[idx]["width"]), int(base.ds[idx]["height"])
            for tile in generate_tiles(w, h, tile_size, overlap):
                self.items.append((idx, tile))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        idx, (x0, y0, x1, y1) = self.items[i]
        item = self.base[idx]
        crop = item["image"].crop((x0, y0, x1, y1))
        return {
            "image": crop,
            "image_id": item["image_id"],
            "target_size": (crop.height, crop.width),
            "offset": (float(x0), float(y0)),
        }


def bucket_of(area: float) -> list[str]:
    return [lbl for lbl, (lo, hi) in AREA_RANGES if lo <= area < hi]


def collision_probe(coco_gt: dict, image_size: int, patch_size: int, zoom: float) -> None:
    """Recall ceiling if each patch can only claim the GT centred in it."""
    by_image: dict[int, list[dict]] = {}
    for ann in coco_gt["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    dims = {im["id"]: (im["width"], im["height"]) for im in coco_gt["images"]}

    total, lost = Counter(), Counter()
    cls_total, cls_lost = Counter(), Counter()
    per_image_counts = []
    for img_id, anns in by_image.items():
        w, h = dims[img_id]
        scale = image_size / max(w, h) * zoom  # model pixels per original pixel
        cells = Counter()
        keys = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            key = (int((x + bw / 2) * scale // patch_size), int((y + bh / 2) * scale // patch_size))
            cells[key] += 1
            keys.append(key)
        per_image_counts.append(len(anns))
        # In a cell with n GTs, n-1 are unrecoverable; charge the loss to every GT
        # in the cell fractionally so buckets/classes are not biased by order.
        for ann, key in zip(anns, keys):
            frac_lost = (cells[key] - 1) / cells[key]
            for lbl in bucket_of(ann["area"]):
                total[lbl] += 1
                lost[lbl] += frac_lost
            cls_total[ann["category_id"]] += 1
            cls_lost[ann["category_id"]] += frac_lost

    patch_px = patch_size / (image_size / 800 * zoom)
    print(f"\n=== Patch-collision ceiling @ zoom {zoom:g}x (1 patch ~ {patch_px:.1f} px on an 800 px DIOR image) ===")
    print(f"  {'bucket':<10s} {'#GT':>8s} {'ceiling':>8s}")
    for lbl, _ in AREA_RANGES:
        if total[lbl]:
            print(f"  {lbl:<10s} {total[lbl]:>8d} {1 - lost[lbl] / total[lbl]:>8.4f}")
    names = {c["id"]: c["name"] for c in coco_gt["categories"]}
    worst = sorted(cls_total, key=lambda c: cls_lost[c] / cls_total[c], reverse=True)[:5]
    print("  worst classes: " + ", ".join(
        f"{names[c]} {1 - cls_lost[c] / cls_total[c]:.3f}" for c in worst
    ))
    counts = np.array(per_image_counts)
    print(f"  GT/image: median {np.median(counts):.0f}, p99 {np.percentile(counts, 99):.0f}, "
          f"max {counts.max()}, images >100 GT: {(counts > 100).sum()}, >300 GT: {(counts > 300).sum()}")


def compact_eval(coco_gt: dict, detections: list[dict], max_dets: int) -> dict[str, float]:
    from faster_coco_eval import COCO, COCOeval_faster

    gt = COCO(coco_gt)
    ev = COCOeval_faster(gt, gt.loadRes(detections), iouType="bbox")
    ev.params.areaRng = [list(r) for _, r in AREA_RANGES]
    ev.params.areaRngLbl = [lbl for lbl, _ in AREA_RANGES]
    ev.params.maxDets = [1, 10, max_dets]
    ev.evaluate()
    ev.accumulate()
    prec, rec = ev.eval["precision"], ev.eval["recall"]
    i50 = int(np.where(np.isclose(ev.params.iouThrs, 0.5))[0][0])

    def mean(a):
        a = a[a > -1]
        return float(a.mean()) if a.size else float("nan")

    out = {}
    for a, (lbl, _) in enumerate(AREA_RANGES):
        out[f"AP {lbl}"] = mean(prec[:, :, :, a, -1])
        out[f"AP50 {lbl}"] = mean(prec[i50, :, :, a, -1])
        out[f"AR {lbl}"] = mean(rec[:, :, a, -1])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-size", default="base", choices=["base", "large"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tile-size", type=int, default=480)
    p.add_argument("--tile-overlap", type=int, default=160)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--collisions-only", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dataset = DiorDetectionDataset(split="test", limit=args.limit)
    coco_gt = dataset.build_coco_gt()
    print(f"DIOR test: {len(dataset)} images, {len(coco_gt['annotations'])} GT boxes")

    image_size, patch_size = (960, 16) if args.model_size == "base" else (1008, 14)
    zoom = 800 / args.tile_size
    for z in (1.0, zoom):
        collision_probe(coco_gt, image_size, patch_size, z)
    if args.collisions_only:
        return

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model_size}_n{len(dataset)}_k{args.top_k}"
    runs = {
        "full": (f"{tag}_full.json", dataset),
        "tiles": (f"{tag}_tile{args.tile_size}o{args.tile_overlap}.json", None),
    }
    model = None
    dets = {}
    for name, (fname, ds) in runs.items():
        path = cache / fname
        if path.exists():
            dets[name] = json.loads(path.read_text())
            print(f"Loaded cached {name} detections from {path}")
            continue
        if model is None:
            print(f"Device: {torch.cuda.get_device_name(0)}")
            model = build_owlv2_model(args.model_size, args.device)
        if ds is None:
            ds = TiledDataset(dataset, args.tile_size, args.tile_overlap)
            print(f"  -> {len(ds)} tiles ({len(ds) / len(dataset):.1f}/image, zoom {zoom:g}x)")
        dets[name] = run_owlv2_inference(
            model, ds, dataset.class_names, device=args.device, score_threshold=0.0,
            top_k=args.top_k, batch_size=args.batch_size, fast_preprocess=True,
            use_autocast=True, num_workers=args.num_workers, desc=f"DIOR {name}",
        )
        path.write_text(json.dumps(dets[name]))

    configs = {
        "full": dets["full"],
        "tiles": merge_tiled_detections(dets["tiles"], args.nms_iou),
        "full+tiles": merge_tiled_detections(dets["full"] + dets["tiles"], args.nms_iou),
    }
    for max_dets in (100, 300):
        results = {name: compact_eval(coco_gt, d, max_dets) for name, d in configs.items()}
        keys = [f"{m} {lbl}" for m in ("AP", "AP50", "AR") for lbl, _ in AREA_RANGES]
        print(f"\n=== DIOR {args.model_size} zoom test, maxDets={max_dets} "
              f"(tiles {args.tile_size}/{args.tile_overlap} -> {zoom:g}x) ===")
        print(f"  {'metric':<16s}" + "".join(f"{n:>12s}" for n in configs))
        for k in keys:
            print(f"  {k:<16s}" + "".join(f"{results[n][k]:>12.4f}" for n in configs))


if __name__ == "__main__":
    main()
