"""Few-shot FLAME vs zero-shot OWLv2 on DIOR (or Fashionpedia).

For every class, ``--support-images`` train images containing it are sampled.
``--mode boxes`` gives FLAME all of those images with all of their ground-truth
boxes. ``--mode active`` instead lets FLAME suggest ``--shots`` proposals per class
from that class's images, and the ground truth answers yes/no (yes at IoU >= 0.5).

The eval split then goes through the model once. The same proposals are scored
zero-shot and by FLAME, and both go through the same post-processing, so the two
reports differ only in the refiner.

    uv run --extra train python tools/test_dior_flame.py --model-size base \
        --support-images 5 --limit 2000
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import replace
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
    class_name_to_query,
    coco_eval_with_custom_sizes,
)

# dataset id, eval split, box format, prompt template
DATASETS = {
    "dior": ("HichTala/dior", "test", "xywh", SATELLITE_QUERY_TEMPLATE),
    "fashionpedia": (FASHIONPEDIA_DATASET_ID, "val", "xyxy", PHOTO_QUERY_TEMPLATE),
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


def main(default_dataset: str = "dior", description: Optional[str] = None) -> None:
    parser = argparse.ArgumentParser(description=description or __doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=default_dataset, choices=sorted(DATASETS))
    parser.add_argument("--model-size", default="base", choices=["base", "large"])
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

    model = OwlV2(args.model_size).eval().to(args.device)
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
    coco_eval_with_custom_sizes(gt, zero_shot, selected, title=f"{args.dataset} zero-shot ({args.model_size})")
    coco_eval_with_custom_sizes(gt, refined, selected, title=(
        f"{args.dataset} FLAME {args.mode} ({args.model_size}, {args.support_images} images/class"
        + (f", {args.shots} shots/class)" if args.mode == "active" else ")")))


if __name__ == "__main__":
    main()
