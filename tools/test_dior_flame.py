"""Few-shot FLAME vs zero-shot OWLv2 on DIOR (or Fashionpedia).

For every class, ``--support-images`` train images containing it are sampled.
``--mode boxes`` gives FLAME all of those images with all of their ground-truth
boxes. ``--mode active`` instead lets FLAME suggest ``--shots`` proposals per class
from that class's images, and the ground truth answers yes/no (yes at IoU >= 0.5).

The eval split then goes through the model once. The same proposals are scored
zero-shot and by FLAME, and both go through the same post-processing, so the two
reports differ only in the refiner.

``--checkpoint`` runs FLAME on top of fine-tuned weights from train_text.py (a
LoRA adapter is merged first; ``--wise-alpha`` interpolates toward pretrained).
On DIOR the per-class table then splits the classes into those DOTA also labels
and the DIOR-only ones, which is the held-out test for a DOTA-trained adapter.

    uv run --extra train python tools/test_dior_flame.py --model-size base \
        --support-images 5 --limit 2000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

from OWLv2torch import OwlV2
from OWLv2torch.torch_version.flame import Flame, select_detections
from ovd_eval import (
    FASHIONPEDIA_DATASET_ID,
    PHOTO_QUERY_TEMPLATE,
    SATELLITE_QUERY_TEMPLATE,
    apply_text_checkpoint,
    class_name_to_query,
    coco_eval_with_custom_sizes,
)

# dataset id, eval split, box format, prompt template
DATASETS = {
    "dior": ("HichTala/dior", "test", "xywh", SATELLITE_QUERY_TEMPLATE),
    "fashionpedia": (FASHIONPEDIA_DATASET_ID, "val", "xyxy", PHOTO_QUERY_TEMPLATE),
}

# DIOR classes DOTA (v1.0 and v1.5) also labels: its small/large vehicle are DIOR's
# vehicle, and its four sports grounds are DIOR's. A DOTA-trained adapter has seen
# these, so a gain on them is mostly class overlap; a gain on the remaining
# DIOR-only classes is the evidence that the adaptation generalizes.
CLASS_GROUPS = {
    "dior": ("shared with DOTA", "DIOR-only", {
        "Airplane", "Ship", "Storage tank", "Vehicle", "Harbor", "Bridge",
        "Baseball field", "Basketball court", "Tennis court", "Ground track field",
    }),
}


def gt_boxes(sample: dict, box_format: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Ground-truth ``xyxy`` boxes and category ids of one dataset sample."""
    boxes = torch.tensor(sample["objects"]["bbox"], dtype=torch.float32).reshape(-1, 4)
    if box_format == "xywh":
        boxes[:, 2:] += boxes[:, :2]
    return boxes, torch.tensor(sample["objects"]["category"], dtype=torch.long)


def coco_gt(dataset, box_format: str, to_query: dict[int, int], names: list[str]) -> dict:
    """COCO ground truth for the evaluated classes, with category ids = query indices."""
    images, annotations = [], []
    for sample in dataset.remove_columns("image"):
        images.append({"id": int(sample["image_id"]), "width": int(sample["width"]),
                       "height": int(sample["height"])})
        boxes, cats = gt_boxes(sample, box_format)
        for box, cat in zip(boxes.tolist(), cats.tolist()):
            w, h = box[2] - box[0], box[3] - box[1]
            if cat in to_query and w > 0 and h > 0:
                annotations.append({"id": len(annotations) + 1, "image_id": images[-1]["id"],
                                    "category_id": to_query[cat], "bbox": [box[0], box[1], w, h],
                                    "area": w * h, "iscrowd": 0})
    categories = [{"id": q, "name": names[cat]} for cat, q in to_query.items()]
    return {"images": images, "annotations": annotations, "categories": categories}


def per_class_ap(gt: dict, detections: list[dict]) -> dict[int, float]:
    """COCO AP@[.5:.95] (all areas, 100 detections) of every category id; NaN without GT."""
    from faster_coco_eval import COCO, COCOeval_faster

    coco_gt = COCO(gt)
    cocoeval = COCOeval_faster(coco_gt, coco_gt.loadRes(detections), iouType="bbox")
    cocoeval.evaluate()
    cocoeval.accumulate()
    precision = cocoeval.eval["precision"][:, :, :, 0, -1]
    aps = {}
    for k, cat in enumerate(cocoeval.params.catIds):
        valid = precision[:, :, k][precision[:, :, k] > -1]
        aps[int(cat)] = float(valid.mean()) if valid.size else float("nan")
    return aps


def report_per_class(dataset: str, selected: list[str], aps: dict[str, dict[int, float]]) -> dict:
    """Print each class's AP per scorer plus group means; return ``{scorer: {per_class, groups}}``."""
    shared_name, other_name, shared = CLASS_GROUPS.get(dataset, (None, None, set()))
    group_of = {name: (shared_name if name in shared else other_name) for name in selected}
    print(f"\n=== per-class AP @[.5:.95] ===")
    print(f"{'class':<26s} {'group':<18s}" + "".join(f" {scorer:>10s}" for scorer in aps))
    for q, name in enumerate(selected):
        print(f"{name:<26s} {group_of[name] or '':<18s}" + "".join(f" {ap[q]:>10.4f}" for ap in aps.values()))
    groups = {}
    if shared_name is not None:
        for group in (shared_name, other_name):
            members = [q for q, name in enumerate(selected) if group_of[name] == group]
            if members:
                groups[group] = members
        print()
        for group, members in groups.items():
            means = [sum(ap[q] for q in members) / len(members) for ap in aps.values()]
            print(f"{group + f' ({len(members)})':<45s}" + "".join(f" {m:>10.4f}" for m in means))
    return {
        scorer: {
            "per_class": {name: ap[q] for q, name in enumerate(selected)},
            "groups": {group: sum(ap[q] for q in members) / len(members) for group, members in groups.items()},
        }
        for scorer, ap in aps.items()
    }


def main(default_dataset: str = "dior", description: Optional[str] = None) -> None:
    parser = argparse.ArgumentParser(description=description or __doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=default_dataset, choices=sorted(DATASETS))
    parser.add_argument("--model-size", default=None, choices=["base", "large"],
                        help="Default: the checkpoint's, else base.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Full, delta or LoRA checkpoint from train_text.py to run FLAME on top of.")
    parser.add_argument("--wise-alpha", type=float, default=1.0,
                        help="WiSE-FT: interpolate --checkpoint toward pretrained (1 = as trained).")
    parser.add_argument("--classes", nargs="*", default=None, help="Class names to evaluate (default: all).")
    parser.add_argument("--mode", default="boxes", choices=["boxes", "active"])
    parser.add_argument("--support-images", type=int, default=5, help="Train images sampled per class.")
    parser.add_argument("--shots", type=int, default=30, help="Labels per class FLAME asks for in --mode active.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate on the first N eval images.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-detections", default=None, help="Write {zero_shot, flame} COCO detections here.")
    parser.add_argument("--save-metrics", default=None, help="Write overall, per-class and group AP here (JSON).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset_id, eval_split, box_format, template = DATASETS[args.dataset]
    train = load_dataset(dataset_id, split="train", cache_dir=args.cache_dir)
    test = load_dataset(dataset_id, split=eval_split, cache_dir=args.cache_dir)
    if args.limit is not None:
        test = test.select(range(min(args.limit, len(test))))
    names: list[str] = test.features["objects"]["category"].feature.names
    unknown = set(args.classes or []) - set(names)
    if unknown:
        raise SystemExit(f"Unknown classes {sorted(unknown)}; known: {names}")
    class_ids = [names.index(n) for n in args.classes] if args.classes else list(range(len(names)))
    to_query = {cat: q for q, cat in enumerate(class_ids)}

    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if args.model_size not in (None, checkpoint.get("model_type")):
            raise SystemExit(f"Checkpoint is {checkpoint.get('model_type')!r}, not --model-size {args.model_size!r}")
        args.model_size = checkpoint.get("model_type")
    args.model_size = args.model_size or "base"
    model = OwlV2(args.model_size).eval().to(args.device)
    if checkpoint is not None:
        apply_text_checkpoint(model, checkpoint, wise_alpha=args.wise_alpha)
        print(f"Loaded {args.checkpoint} ({checkpoint['format']}, wise_alpha={args.wise_alpha})")
    flame = Flame(model, [class_name_to_query(names[c], template) for c in class_ids],
                  batch_size=args.batch_size)

    # Support images: --support-images per class, pooled, with every selected class labelled.
    rng = random.Random(args.seed)
    by_class: dict[int, list[int]] = {c: [] for c in class_ids}
    for i, objects in enumerate(train["objects"]):
        for c in set(objects["category"]) & by_class.keys():
            by_class[c].append(i)
    chosen = {c: rng.sample(by_class[c], min(args.support_images, len(by_class[c]))) for c in class_ids}
    support = sorted({i for c in class_ids for i in chosen[c]})
    samples = [train[i] for i in support]
    labelled = []
    for s in samples:
        boxes, cats = gt_boxes(s, box_format)
        keep = torch.tensor([c in to_query for c in cats.tolist()], dtype=torch.bool)
        labelled.append((boxes[keep], torch.tensor([to_query[c] for c in cats[keep].tolist()], dtype=torch.long)))

    start = time.perf_counter()
    if args.mode == "boxes":
        for s, (boxes, labels) in zip(tqdm(samples, desc="examples"), labelled):
            flame.add_example(s["image"], boxes, labels)
    else:
        # As in the paper, each class's questions come from images that contain it.
        position = {i: n for n, i in enumerate(support)}
        candidates = []
        for c in class_ids:
            pool = [position[i] for i in sorted(chosen[c])]
            suggested = flame.suggest([samples[n]["image"] for n in pool], k=args.shots)
            candidates += [replace(cand, image=pool[cand.image]) for cand in suggested if cand.label == to_query[c]]
        answers = []
        for cand in candidates:
            boxes, labels = labelled[cand.image]
            own = boxes[labels == cand.label]
            answers.append(bool(len(own)) and box_iou(torch.tensor([cand.box]), own).max().item() >= 0.5)
        flame.add_labels(candidates, answers)
        print(f"answered {len(answers)} suggestions: {sum(answers)} yes")
    flame.fit()
    print(f"FLAME adapted on {len(samples)} images in {time.perf_counter() - start:.1f}s")

    zero_shot, refined = [], []
    loader = DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=lambda b: b)
    with torch.inference_mode():
        for batch in tqdm(loader, desc="eval"):
            proposals = flame.proposals([s["image"] for s in batch])
            for s, (embeds, logits, objectness, boxes) in zip(batch, proposals):
                plain = torch.sigmoid(logits) * torch.sigmoid(objectness)[:, None]
                for out, scores in ((zero_shot, plain), (refined, flame.scores(embeds, logits, objectness))):
                    for box, score, label in zip(*select_detections(boxes, scores, threshold=0.0)):
                        x1, y1, x2, y2 = box.tolist()
                        out.append({"image_id": int(s["image_id"]), "category_id": int(label),
                                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})

    if args.save_detections:
        with open(args.save_detections, "w") as f:
            json.dump({"zero_shot": zero_shot, "flame": refined}, f)
    gt = coco_gt(test, box_format, to_query, names)
    selected = [names[c] for c in class_ids]
    weights = args.model_size + (f" + {args.checkpoint.parent.name}" if args.checkpoint else "")
    overall = {
        "zero-shot": coco_eval_with_custom_sizes(gt, zero_shot, selected, title=f"{args.dataset} zero-shot ({weights})"),
        "FLAME": coco_eval_with_custom_sizes(gt, refined, selected, title=(
            f"{args.dataset} FLAME {args.mode} ({weights}, {args.support_images} images/class"
            + (f", {args.shots} shots/class)" if args.mode == "active" else ")"))),
    }
    breakdown = report_per_class(args.dataset, selected, {
        "zero-shot": per_class_ap(gt, zero_shot), "FLAME": per_class_ap(gt, refined)})

    if args.save_metrics:
        def finite(value):
            if isinstance(value, dict):
                return {k: finite(v) for k, v in value.items()}
            return None if isinstance(value, float) and math.isnan(value) else value

        payload = {
            "dataset": args.dataset, "model_size": args.model_size,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "wise_alpha": args.wise_alpha if args.checkpoint else None,
            "mode": args.mode, "support_images": args.support_images, "shots": args.shots,
            "seed": args.seed, "limit": args.limit, "classes": selected,
            "scorers": {scorer: {"overall": overall[scorer], **breakdown[scorer]} for scorer in overall},
        }
        path = Path(args.save_metrics)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(finite(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Wrote metrics to {path}")


if __name__ == "__main__":
    main()
