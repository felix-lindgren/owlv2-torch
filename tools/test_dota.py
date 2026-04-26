"""Evaluate OWLv2 on local DOTAv1.5 images and labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from OWLv2torch import OwlV2
from ovd_eval import (
    DotaDetectionDataset,
    coco_eval_with_custom_sizes,
    run_owlv2_inference,
    save_detections,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", default="large", choices=["base", "large"])
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/dotav1_5"),
        help="Folder containing images/ and labels/ subfolders.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally cap the number of evaluation images.",
    )
    parser.add_argument(
        "--include-difficult",
        action="store_true",
        help="Keep DOTA difficult GT boxes in the eval.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Image/tile crops per forward pass. Default: 32 for base, 8 for large.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fast-preprocess", action="store_true")
    parser.add_argument("--no-autocast", dest="autocast", action="store_false")
    parser.set_defaults(autocast=True)
    parser.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="If >0, run sliding-window inference with this square tile size.",
    )
    parser.add_argument("--tile-overlap", type=int, default=200)
    parser.add_argument("--save-detections", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = 32 if args.model_size == "base" else 8

    print(f"Loading OwlV2 ({args.model_size}) on {args.device}...")
    model = OwlV2(args.model_size).eval().to(args.device)
    if str(args.device).startswith("cuda") and not args.fast_preprocess:
        print(
            "[WARN] Using exact scipy image preprocessing. This is CPU-bound and can "
            "make GPU utilization look bursty/low. Add --fast-preprocess for higher throughput."
        )
    print(
        f"Inference settings: batch_size={args.batch_size}, "
        f"autocast={args.autocast}, fast_preprocess={args.fast_preprocess}, "
        f"num_workers={args.num_workers}"
    )

    print(f"Scanning {args.data_root}...")
    dataset = DotaDetectionDataset(
        data_root=args.data_root,
        limit=args.limit,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        include_difficult=args.include_difficult,
    )
    print(f"  -> {len(dataset.samples)} image/label pairs")
    print(f"  -> {len(dataset)} inference crops")
    print(f"  -> {len(dataset.class_names)} classes: {dataset.class_names}")

    coco_gt_dict = dataset.build_coco_gt()
    n_gt = len(coco_gt_dict["annotations"])
    print(
        f"  -> {n_gt} GT boxes "
        f"({'incl.' if args.include_difficult else 'excl.'} difficult)"
    )

    detections = run_owlv2_inference(
        model,
        dataset,
        dataset.class_names,
        device=args.device,
        score_threshold=args.score_threshold,
        top_k=args.top_k,
        batch_size=args.batch_size,
        fast_preprocess=args.fast_preprocess,
        use_autocast=args.autocast,
        num_workers=args.num_workers,
        desc=f"DOTA inference (tile={args.tile_size or 'full'})",
    )

    if args.save_detections:
        save_detections(args.save_detections, detections)
        print(f"Wrote {len(detections)} detections to {args.save_detections}")

    coco_eval_with_custom_sizes(coco_gt_dict, detections, dataset.class_names, title="DOTAv1.5")


if __name__ == "__main__":
    main()
