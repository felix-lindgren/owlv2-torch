"""Shared evaluator for the UAV COCO benchmarks under /mnt/datasets/uav.

MAVREC, Songdo and the WiSARD sample all ship plain COCO-detection JSON, so the
only per-dataset work is knowing which annotation file pairs with which image
root, what prompt suits the sensor, and how hard the frames need to be tiled.
That lives in ``SPLITS`` below; everything else is ``ovd_eval``.

These are 4K-ish frames holding 15-60 px objects. Resized whole to the model's
960/1008 px input a Songdo car is ~15 px across, so the tiled defaults are not a
tuning knob but the difference between a real number and a floor of zero. Pass
``--tile-size 0`` to see the untiled baseline.

Entry points: ``test_mavrec.py``, ``test_songdo.py``, ``test_wisard.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from ovd_eval import (
    AERIAL_QUERY_TEMPLATE,
    PHOTO_QUERY_TEMPLATE,
    CocoFileDetectionDataset,
    apply_text_checkpoint,
    build_owlv2_model,
    checkpoint_touches_vision,
    coco_eval_with_custom_sizes,
    merge_tiled_detections,
    run_owlv2_inference,
    save_detections,
)

UAV_ROOT = Path(os.environ.get("UAV_DATASETS_ROOT", "/mnt/datasets/uav"))

# Alias banks are opt-in (--alias-pooling). Each class keeps its own name first
# so the pooled score can only ever go up relative to the plain prompt.
MAVREC_ALIASES = {
    "tram": ["tram", "light rail train"],
    "van": ["van", "minivan"],
    "truck": ["truck", "lorry"],
    "streetlight": ["streetlight", "street lamp", "lamp post"],
    "traffic_light": ["traffic light", "traffic signal"],
    "other": ["other object", "vehicle"],
}
SONGDO_ALIASES = {
    "car": ["car", "passenger car"],
    "truck": ["truck", "lorry"],
    "motorcycle": ["motorcycle", "scooter"],
}
WISARD_ALIASES = {"human": ["human", "person"]}


@dataclass(frozen=True)
class UavSplit:
    """One annotation file, its image root, and the defaults it wants."""

    ann_file: str
    image_root: str
    title: str
    query_template: str = AERIAL_QUERY_TEMPLATE
    tile_size: int = 0
    tile_overlap: int = 256
    aliases: dict[str, list[str]] = field(default_factory=dict)
    note: str = ""

    def resolve(self) -> tuple[Path, Path]:
        ann = UAV_ROOT / self.ann_file
        root = UAV_ROOT / self.image_root
        if not ann.exists():
            raise FileNotFoundError(
                f"Annotation file {ann} not found. Set UAV_DATASETS_ROOT if the "
                f"datasets live somewhere other than {UAV_ROOT}."
            )
        if not root.is_dir():
            raise FileNotFoundError(f"Image root {root} not found.")
        return ann, root


# MAVREC pairs a drone view and a time-synchronised ground camera over the same
# scene; the ground stream is not aerial, so it gets the plain photo prompt.
# Only train_subset is on disk locally - the full 8,605-image aerial_train split
# has annotations but no images (see MAVREC/RESUME_TRAIN_DOWNLOAD.md).
SPLITS: dict[str, dict[str, UavSplit]] = {
    "mavrec": {
        "aerial-val": UavSplit(
            ann_file="MAVREC/supervised_annotations/aerial/aerial_valid.json",
            image_root="MAVREC/val/aerial",
            title="MAVREC aerial val",
            query_template=AERIAL_QUERY_TEMPLATE,
            tile_size=1024,
            aliases=MAVREC_ALIASES,
            note="538 images / 42,927 boxes, median side 34 px at 2704x1520.",
        ),
        "ground-val": UavSplit(
            ann_file="MAVREC/supervised_annotations/ground/ground_val.json",
            image_root="MAVREC/val/ground",
            title="MAVREC ground val",
            query_template=PHOTO_QUERY_TEMPLATE,
            tile_size=1024,
            aliases=MAVREC_ALIASES,
            note="Ground-level camera, same 538 scenes as aerial-val.",
        ),
        "aerial-train-subset": UavSplit(
            ann_file="MAVREC/train_subset/annotations/aerial_train_subset.json",
            image_root="MAVREC/train_subset/aerial",
            title="MAVREC aerial train subset",
            query_template=AERIAL_QUERY_TEMPLATE,
            tile_size=1024,
            aliases=MAVREC_ALIASES,
        ),
        "ground-train-subset": UavSplit(
            ann_file="MAVREC/train_subset/annotations/ground_train_subset.json",
            image_root="MAVREC/train_subset/ground",
            title="MAVREC ground train subset",
            query_template=PHOTO_QUERY_TEMPLATE,
            tile_size=1024,
            aliases=MAVREC_ALIASES,
        ),
    },
    "songdo": {
        "test": UavSplit(
            ann_file="songdo/test/coco_annotations.json",
            image_root="songdo/test/images",
            title="Songdo test",
            tile_size=1280,
            aliases=SONGDO_ALIASES,
            note="1,084 images / 55,528 boxes at 3840x2160, ~51 objects per image.",
        ),
        "train": UavSplit(
            ann_file="songdo/train/coco_annotations.json",
            image_root="songdo/train/images",
            title="Songdo train",
            tile_size=1280,
            aliases=SONGDO_ALIASES,
            note="4,335 images - use --limit while iterating.",
        ),
    },
    # WiSARD's two streams see the same humans at wildly different scales: 17 px
    # in the 640x512 thermal frame, 67 px in the 4K visual one. IR frames are
    # already smaller than the model input, so tiling them would only upsample
    # noise; VIS needs it.
    "wisard": {
        "ir": UavSplit(
            ann_file="WiSARD/coco/210417_mterie_enterprise_ir_0004.json",
            image_root="WiSARD/210417_MtErie_Enterprise_IR_0004",
            title="WiSARD Mt Erie IR",
            query_template="a thermal aerial image of a {name}",
            tile_size=0,
            aliases=WISARD_ALIASES,
            note="264 thermal frames / 1,006 boxes at 640x512, median side 17 px.",
        ),
        "vis": UavSplit(
            ann_file="WiSARD/coco/210417_mterie_enterprise_vis_0003.json",
            image_root="WiSARD/210417_MtErie_Enterprise_VIS_0003",
            title="WiSARD Mt Erie VIS",
            query_template=AERIAL_QUERY_TEMPLATE,
            tile_size=1280,
            aliases=WISARD_ALIASES,
            note="264 visual frames / 1,022 boxes at 3840x2160, median side 67 px.",
        ),
    },
}

DEFAULT_SPLIT = {"mavrec": "aerial-val", "songdo": "test", "wisard": "ir"}


def _json_safe(value):
    raise TypeError(f"Cannot serialise {value!r}")


def arm_label(tile_size: int) -> str:
    """Name an evaluation arm the way the comparison table should read."""
    return "downscale" if tile_size == 0 else f"tile{tile_size}"


def arm_path(path: str, label: str, multiple_arms: bool) -> str:
    """Keep per-arm outputs from overwriting each other."""
    if not multiple_arms:
        return path
    target = Path(path)
    return str(target.with_name(f"{target.stem}.{label}{target.suffix}"))


def build_parser(default_dataset: Optional[str], description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset",
        choices=sorted(SPLITS),
        default=default_dataset,
        required=default_dataset is None,
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split/stream. Default: "
        + ", ".join(f"{k}={v}" for k, v in sorted(DEFAULT_SPLIT.items())),
    )
    parser.add_argument("--model-size", default=None, choices=["base", "large"])
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Full or delta checkpoint written by prototype_train/train_text.py.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap evaluated images.")
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help=(
            "Evaluate only these category names. mAP averages over whatever set "
            "you pass, so a subset score is not comparable with the full one."
        ),
    )
    parser.add_argument(
        "--tile-size", type=int, nargs="+", default=None,
        help=(
            "One or more sliding-window sizes to evaluate; 0 means whole frames. "
            "Default is the split's own pair, '0 <tile>', so every run reports the "
            "naive-downscale baseline next to the tiled score."
        ),
    )
    parser.add_argument("--tile-overlap", type=int, default=None)
    parser.add_argument(
        "--tile-nms-iou", type=float, default=0.6,
        help="IoU for cross-tile NMS; 0 disables. Ignored without tiling.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=300, help="Kept detections per crop.")
    parser.add_argument(
        "--eval-max-detections", type=int, default=100,
        help="COCO maxDets. 100 is standard AP; these scenes hold 50-80 objects.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Crops per forward pass. Default: 32 for base, 8 for large.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--tensorrt", action="store_true",
        help=(
            "Run the vision tower through TensorRT. The engine is exported and "
            "built on first use (~1-2 min) and cached under --trt-engine-dir."
        ),
    )
    parser.add_argument(
        "--trt-engine", type=Path, default=None,
        help="Explicit .engine path; default is <dir>/owlv2_vis_<size>.engine.",
    )
    parser.add_argument("--trt-engine-dir", type=Path, default=Path("trt_engines"))
    parser.add_argument(
        "--trt-allow-fallback", action="store_true",
        help="Run in torch instead of failing when no engine can be loaded.",
    )
    parser.add_argument("--fast-preprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-autocast", dest="autocast", action="store_false")
    parser.set_defaults(autocast=True)
    parser.add_argument(
        "--query-template", default=None,
        help="Prompt template containing {name}; defaults to the split's own.",
    )
    parser.add_argument(
        "--alias-pooling", action="store_true",
        help="Max-pool a few hand-written synonyms per class into its score.",
    )
    parser.add_argument("--save-detections", default=None)
    parser.add_argument("--save-metrics", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(default_dataset: Optional[str] = None, description: str = __doc__) -> None:
    args = build_parser(default_dataset, description).parse_args()

    split_name = args.split or DEFAULT_SPLIT[args.dataset]
    splits = SPLITS[args.dataset]
    if split_name not in splits:
        raise SystemExit(
            f"Unknown split {split_name!r} for {args.dataset}; "
            f"choose from {sorted(splits)}"
        )
    split = splits[split_name]
    ann_file, image_root = split.resolve()

    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        checkpoint_model_size = checkpoint.get("model_type")
        if checkpoint_model_size not in {"base", "large"}:
            raise ValueError(f"Checkpoint has invalid model_type={checkpoint_model_size!r}")
        if args.model_size is not None and checkpoint_model_size != args.model_size:
            raise ValueError(
                f"Checkpoint model_type={checkpoint_model_size!r}, "
                f"but --model-size={args.model_size!r}"
            )
        args.model_size = checkpoint_model_size
        if args.tensorrt and checkpoint_touches_vision(checkpoint):
            raise SystemExit(
                f"{args.checkpoint} fine-tunes the vision tower, but the TensorRT "
                "engine is exported from the pretrained weights before any "
                "checkpoint is applied - the run would silently measure the "
                "pretrained tower. Drop --tensorrt for this checkpoint."
            )
    args.model_size = args.model_size or "large"
    if args.batch_size is None:
        args.batch_size = 32 if args.model_size == "base" else 8
    if args.top_k is not None and args.top_k < args.eval_max_detections:
        raise ValueError("--top-k cannot be smaller than --eval-max-detections")

    if args.tile_size is None:
        # The naive whole-frame downscale is the number every tiled score has to
        # beat, so it is always an arm rather than something you opt into.
        requested_tiles = [0, split.tile_size]
    else:
        requested_tiles = list(args.tile_size)
    tile_sizes: list[int] = []
    for size in requested_tiles:
        if max(0, size) not in tile_sizes:
            tile_sizes.append(max(0, size))
    tile_overlap = split.tile_overlap if args.tile_overlap is None else args.tile_overlap
    query_template = args.query_template or split.query_template

    print(f"{split.title}  ({ann_file})")
    if split.note:
        print(f"  {split.note}")
    backend = "TensorRT vision tower" if args.tensorrt else "torch"
    print(f"Loading OwlV2 ({args.model_size}, {backend}) on {args.device}...")
    model = build_owlv2_model(
        args.model_size,
        args.device,
        tensorrt=args.tensorrt,
        engine_path=args.trt_engine,
        engine_dir=args.trt_engine_dir,
        batch_size=args.batch_size,
        allow_torch_fallback=args.trt_allow_fallback,
    )
    if checkpoint is not None:
        apply_text_checkpoint(model, checkpoint)
        print(f"Loaded fine-tuned weights from {args.checkpoint}")
    if str(args.device).startswith("cuda") and not args.fast_preprocess:
        print(
            "[WARN] Using exact scipy image preprocessing. This is CPU-bound and can "
            "make GPU utilization look bursty/low. Drop --no-fast-preprocess for higher throughput."
        )
    print(
        f"Inference settings: batch_size={args.batch_size}, "
        f"autocast={args.autocast}, fast_preprocess={args.fast_preprocess}, "
        f"num_workers={args.num_workers}"
    )
    print(f"Evaluation prompt: {query_template!r}")
    print(
        f"Detection policy: inference top_k={args.top_k} per crop; "
        f"COCO maxDets={args.eval_max_detections}"
        + (" (standard)" if args.eval_max_detections == 100 else " (non-standard dense metric)")
    )
    print(f"Arms: {', '.join(arm_label(size) for size in tile_sizes)}")

    arms: list[dict] = []
    for tile_size in tile_sizes:
        label = arm_label(tile_size)
        title = f"{split.title} [{label}]"
        dataset = CocoFileDetectionDataset(
            ann_file=ann_file,
            image_root=image_root,
            limit=args.limit,
            class_names=args.classes,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        print(f"\n--- {title} ---")
        print(f"  -> {len(dataset.images)} images, {len(dataset)} inference crops")
        print(f"  -> {len(dataset.class_names)} classes: {dataset.class_names}")
        if args.classes is not None:
            print(
                "  [WARN] Class subset in use: this mAP is not comparable with the "
                "full-vocabulary score."
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
            desc=f"{title} inference",
            query_template=query_template,
            query_aliases=(split.aliases if args.alias_pooling else None),
        )

        if tile_size and args.tile_nms_iou > 0:
            before = len(detections)
            detections = merge_tiled_detections(detections, args.tile_nms_iou)
            print(
                f"Cross-tile NMS @IoU {args.tile_nms_iou}: "
                f"{before} -> {len(detections)} detections"
            )

        if args.save_detections:
            path = arm_path(args.save_detections, label, len(tile_sizes) > 1)
            save_detections(path, detections)
            print(f"Wrote {len(detections)} detections to {path}")

        metrics = coco_eval_with_custom_sizes(
            dataset.build_coco_gt(),
            detections,
            dataset.class_names,
            title=title,
            max_detections=args.eval_max_detections,
        )
        metrics = {
            key: (None if value != value else value) for key, value in metrics.items()
        }
        arms.append({
            "label": label,
            "tile_size": tile_size,
            "tile_overlap": tile_overlap if tile_size else None,
            "tile_nms_iou": args.tile_nms_iou if tile_size else None,
            "crops": len(dataset),
            "detections": len(detections),
            "metrics": metrics,
        })

    if len(arms) > 1:
        print(f"\n=== {split.title}: arm comparison ===")
        header = f"{'arm':<16s} {'crops':>8s} {'mAP':>8s} {'mAP@50':>8s} {'small':>8s} {'medium':>8s} {'large':>8s}"
        print(header)
        for arm in arms:
            m = arm["metrics"]
            values = "".join(
                f" {m.get(key, float('nan')):>8.4f}"
                for key in ("map", "map_50", "map_small", "map_medium", "map_large")
            )
            print(f"{arm['label']:<16s} {arm['crops']:>8d}{values}")

    if args.save_metrics:
        payload = {
            "dataset": args.dataset,
            "split": split_name,
            "ann_file": str(ann_file),
            "image_root": str(image_root),
            "model_size": args.model_size,
            "tensorrt": args.tensorrt and getattr(model, "trt", None) is not None,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "query_template": query_template,
            "alias_pooling": args.alias_pooling,
            "classes": args.classes,
            "limit": args.limit,
            "inference_top_k": args.top_k,
            "eval_max_detections": args.eval_max_detections,
            "arms": arms,
        }
        path = Path(args.save_metrics)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            # An absent area bucket comes back as NaN, which json.dump would
            # write as a bare NaN token that no strict JSON reader accepts.
            json.dump(payload, handle, indent=2, allow_nan=False, default=_json_safe)
            handle.write("\n")
        print(f"Wrote metrics to {path}")


if __name__ == "__main__":
    main()
