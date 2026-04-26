"""Evaluate OWLv2 on the DIOR test split."""

from __future__ import annotations

import argparse

import torch

from OWLv2torch import OwlV2
from ovd_eval import (
    DiorDetectionDataset,
    coco_eval_with_custom_sizes,
    run_owlv2_inference,
    save_detections,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", default="large", choices=["base", "large"])
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally cap the number of evaluation samples.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Images per forward pass. Default: 32 for base, 8 for large.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fast-preprocess", action="store_true")
    parser.add_argument("--no-autocast", dest="autocast", action="store_false")
    parser.set_defaults(autocast=True)
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

    print(f"Loading HichTala/dior split={args.split}...")
    dataset = DiorDetectionDataset(split=args.split, limit=args.limit)
    print(f"  -> {len(dataset)} samples")
    print(f"  -> {len(dataset.class_names)} classes: {dataset.class_names}")

    coco_gt_dict = dataset.build_coco_gt()
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
        desc="DIOR inference",
    )

    if args.save_detections:
        save_detections(args.save_detections, detections)
        print(f"Wrote {len(detections)} detections to {args.save_detections}")

    coco_eval_with_custom_sizes(coco_gt_dict, detections, dataset.class_names, title="DIOR test set")


if __name__ == "__main__":
    main()
