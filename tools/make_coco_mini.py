"""Freeze a small, class-covering COCO val2017 subset for general-domain tracking.

Fine-tuning on a narrow domain (fashion, remote sensing, ...) can erode OWLv2's
open-vocabulary ability. This subset is a cheap probe for that: score every
checkpoint on the same few hundred COCO images with the 80 COCO class names and
watch how far mAP falls from the pretrained model.

Selection is deterministic for a given ``--seed``:

* Images with any ``iscrowd`` annotation are excluded. The in-training evaluator
  (``prototype_train/train.py:coco_eval``) has no crowd handling, so a crowd
  region would count as a missed object and the number would drift from what
  standard COCO evaluation reports for the same detections.
* Classes are visited from rarest to most common, and images containing each
  class are added until it appears in ``--min-images-per-class`` images. The
  rest of the budget is filled at random, so common classes keep roughly
  their natural frequency.

Example:

    uv run python tools/make_coco_mini.py \
        --annotations /mnt/datasets/coco/annotations/instances_val2017.json \
        --output /mnt/datasets/coco/annotations/instances_val2017_mini500.json

Then pass to ``prototype_train/train_text.py``:

    --general-val-annotations /mnt/datasets/coco/annotations/instances_val2017_mini500.json
    --general-val-images /mnt/datasets/coco/val2017

or score a finished checkpoint with ``tools/ovd_eval.py --dataset coco``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def select_images(
    coco: dict, num_images: int, min_images_per_class: int, seed: int
) -> list[int]:
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)

    eligible = sorted(
        image_id
        for image_id, anns in anns_by_image.items()
        if not any(ann.get("iscrowd", 0) for ann in anns)
    )
    random.Random(seed).shuffle(eligible)

    images_by_class: dict[int, list[int]] = defaultdict(list)
    for image_id in eligible:
        for category_id in {int(ann["category_id"]) for ann in anns_by_image[image_id]}:
            images_by_class[category_id].append(image_id)

    selected: list[int] = []
    selected_set: set[int] = set()
    class_coverage: Counter = Counter()

    def add(image_id: int) -> None:
        selected.append(image_id)
        selected_set.add(image_id)
        for category_id in {int(ann["category_id"]) for ann in anns_by_image[image_id]}:
            class_coverage[category_id] += 1

    for category_id in sorted(images_by_class, key=lambda c: (len(images_by_class[c]), c)):
        for image_id in images_by_class[category_id]:
            if class_coverage[category_id] >= min_images_per_class:
                break
            if image_id not in selected_set:
                add(image_id)

    if len(selected) > num_images:
        raise ValueError(
            f"Covering every class {min_images_per_class}x already needs "
            f"{len(selected)} images; raise --num-images or lower --min-images-per-class"
        )
    for image_id in eligible:
        if len(selected) >= num_images:
            break
        if image_id not in selected_set:
            add(image_id)
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--annotations", type=Path,
        default=Path("/mnt/datasets/coco/annotations/instances_val2017.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--min-images-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with args.annotations.open(encoding="utf-8") as handle:
        coco = json.load(handle)

    image_ids = select_images(
        coco, args.num_images, args.min_images_per_class, args.seed
    )
    keep = set(image_ids)
    subset = {
        "info": {
            **coco.get("info", {}),
            "description": (
                f"COCO val2017 general-domain mini set: {len(image_ids)} crowd-free "
                f"images, every class in >= {args.min_images_per_class} images"
            ),
            "source": str(args.annotations),
            "seed": args.seed,
            "num_images": args.num_images,
            "min_images_per_class": args.min_images_per_class,
        },
        "licenses": coco.get("licenses", []),
        "images": [img for img in coco["images"] if int(img["id"]) in keep],
        "categories": coco["categories"],
        "annotations": [
            ann for ann in coco["annotations"] if int(ann["image_id"]) in keep
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(subset, handle)

    names = {int(cat["id"]): cat["name"] for cat in coco["categories"]}
    instances = Counter(int(ann["category_id"]) for ann in subset["annotations"])
    image_counts = Counter(
        category_id
        for image_id in keep
        for category_id in {
            int(ann["category_id"])
            for ann in subset["annotations"]
            if int(ann["image_id"]) == image_id
        }
    )
    areas = Counter(
        "small" if ann["area"] < 32 ** 2 else "medium" if ann["area"] < 96 ** 2 else "large"
        for ann in subset["annotations"]
    )
    rarest = sorted(names, key=lambda c: (image_counts[c], c))[:5]
    print(f"Wrote {args.output}")
    print(
        f"  {len(subset['images'])} images, {len(subset['annotations'])} annotations, "
        f"{len(instances)}/{len(names)} classes present"
    )
    print(
        "  instances by area: "
        + ", ".join(f"{bucket}={areas[bucket]}" for bucket in ("small", "medium", "large"))
    )
    print(
        "  fewest images per class: "
        + ", ".join(f"{names[c]}={image_counts[c]}" for c in rarest)
    )


if __name__ == "__main__":
    main()
