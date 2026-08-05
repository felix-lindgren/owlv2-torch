"""Visualise and score a ``train_text.py`` checkpoint on a COCO validation split.

Written for the Fashionpedia runs in ``text_checkpoints/``, but nothing here is
Fashionpedia-specific: the dataset paths, class names, evaluation prompts and
scoring protocol all come out of the checkpoint's stored config, so any COCO-style
run made by ``prototype_train/train_text.py`` works.

Two outputs land in ``--output-dir``:

* ``panels/`` — one figure per selected image, predictions and ground truth
  overlaid and coloured by match status, so misses and false positives are
  legible at a glance. ``--select worst`` ranks images by F1 first, which is
  usually the fastest way to see what a run is actually failing at.
* ``metrics.txt`` and ``per_class_ap.png`` — torchmetrics numbers including
  per-class AP. The metric pass deliberately mirrors the protocol
  ``train_text.py`` evaluates with (no NMS, ``--eval-top-k`` detections,
  ``--confidence-threshold`` 0.001), so the headline ``map`` here is directly
  comparable to the numbers in ``docs/run-plan.md``.

The panels use a different, much stricter set of thresholds — drawing 100 boxes
per image at score 0.001 is unreadable — so panel counts will not reconcile with
the mAP figure. That is intentional; they answer different questions.

Example:

    uv run python tools/visualize_text_checkpoint.py \
        --checkpoint fashionpedia_large_C1_vb6_20260803 \
        --select worst --num-images 12
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.detection import MeanAveragePrecision
from torchvision.datasets import CocoDetection
from torchvision.ops import batched_nms, box_iou
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OWLv2torch.torch_version.owlv2 import OwlV2
from prototype_train.train import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    coco_class_names,
    coco_collate_fn,
)
from prototype_train.train_text import (
    TextQueryDetector,
    evaluation_prompts,
    load_text_checkpoint,
)


# Status palette. Match status is also encoded by line style and by a glyph in
# every label, because good/critical sit at CVD deltaE 4.1 — the colour alone
# does not survive deuteranopia.
STATUS_STYLES = {
    "true_positive": {"color": "#0ca30c", "linestyle": "solid", "glyph": "✓"},
    "false_positive": {"color": "#d03b3b", "linestyle": (0, (6, 2, 1, 2)), "glyph": "✗"},
    "missed": {"color": "#fab219", "linestyle": (0, (5, 3)), "glyph": "!"},
    "matched_gt": {"color": "#f0efec", "linestyle": (0, (1, 2)), "glyph": ""},
}
STATUS_LABELS = {
    "true_positive": "prediction, matched (✓)",
    "false_positive": "prediction, unmatched (✗)",
    "missed": "ground truth, missed (!)",
    "matched_gt": "ground truth, matched",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
HALO = "#0b0b0b"
BAR_COLOR = "#2a78d6"


def resolve_checkpoint_path(raw: str, default_root: Path) -> Path:
    """Accept a file, a run directory, or a bare run name under ``default_root``."""
    candidates = [Path(raw), default_root / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            for name in ("best.pth", "last.pth", "final.pth"):
                if (candidate / name).is_file():
                    return candidate / name
            raise FileNotFoundError(f"No best/last/final .pth inside {candidate}")
    raise FileNotFoundError(f"No checkpoint at {raw} or {default_root / raw}")


def config_default(config: dict, key: str, fallback=None):
    value = config.get(key)
    return fallback if value is None else value


def build_detector(
    checkpoint: dict, class_names: list[str], prompt_template: str | None, device
) -> tuple[TextQueryDetector, list[str]]:
    model = OwlV2(checkpoint["model_type"])
    load_text_checkpoint(model, checkpoint["_path"])
    if prompt_template is None:
        prompts = list(checkpoint["evaluation_prompts"])
    else:
        prompts = evaluation_prompts(class_names, prompt_template)
    return TextQueryDetector(model, prompts).to(device).eval(), prompts


def build_zero_shot_detector(
    model_type: str, prompts: list[str], device
) -> TextQueryDetector:
    """The same prompts against stock OWLv2 weights, for a before/after panel."""
    return TextQueryDetector(OwlV2(model_type), prompts).to(device).eval()


def ground_truth_xyxy(target: dict, image_size: float) -> torch.Tensor:
    """Normalised cxcywh in padded-square space -> absolute xyxy, as ``coco_eval`` does."""
    boxes = target["boxes"]
    if boxes.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    centre_x = boxes[:, 0] * image_size
    centre_y = boxes[:, 1] * image_size
    width = boxes[:, 2] * image_size
    height = boxes[:, 3] * image_size
    return torch.stack(
        [
            centre_x - width / 2,
            centre_y - height / 2,
            centre_x + width / 2,
            centre_y + height / 2,
        ],
        dim=1,
    )


@torch.no_grad()
def predict_batches(
    detector: TextQueryDetector,
    loader: DataLoader,
    device: torch.device,
    *,
    score_threshold: float,
    top_k: int | None,
    nms_iou: float | None,
    score_with_objectness: bool,
    amp: bool,
    keep_images: bool,
    description: str,
):
    """Yield one dict per image: predictions, ground truth, and optionally the input.

    Scores are ``sigmoid(class) * sigmoid(objectness)`` and labels are the
    per-box argmax, matching ``prototype_train.train.coco_eval`` so that anything
    computed here lines up with the training-time metric.
    """
    detector.eval()
    for batch in tqdm(loader, desc=description):
        pixel_values = batch["images"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            outputs = detector(pixel_values)
        logits, objectness_logits, raw_boxes = outputs[0], outputs[1], outputs[2]

        batch_size, _, height, width = pixel_values.shape
        target_sizes = torch.tensor(
            [(height, width)] * batch_size, device=device, dtype=torch.float32
        )
        boxes = detector.owl.postprocess_boxes(raw_boxes, target_sizes)

        class_probabilities = torch.sigmoid(logits.float())
        if score_with_objectness:
            class_probabilities = class_probabilities * torch.sigmoid(
                objectness_logits.float()
            ).unsqueeze(-1)
        scores, labels = class_probabilities.max(dim=-1)

        for index in range(batch_size):
            keep = scores[index] > score_threshold
            image_boxes = boxes[index][keep]
            image_scores = scores[index][keep]
            image_labels = labels[index][keep]
            if nms_iou is not None and image_scores.numel():
                kept = batched_nms(
                    image_boxes.float(), image_scores, image_labels, nms_iou
                )
                image_boxes = image_boxes[kept]
                image_scores = image_scores[kept]
                image_labels = image_labels[kept]
            if top_k is not None and image_scores.numel() > top_k:
                kept = torch.topk(image_scores, top_k).indices
                image_boxes = image_boxes[kept]
                image_scores = image_scores[kept]
                image_labels = image_labels[kept]

            target = batch["targets"][index]
            yield {
                "boxes": image_boxes.float().cpu(),
                "scores": image_scores.float().cpu(),
                "labels": image_labels.cpu().to(torch.int64),
                "gt_boxes": ground_truth_xyxy(target, float(height)),
                "gt_labels": target["labels"].cpu().to(torch.int64),
                "image": pixel_values[index].cpu() if keep_images else None,
            }


def match_predictions(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy COCO-style matching by descending score, same class, IoU >= threshold."""
    pred_matched = torch.zeros(pred_boxes.shape[0], dtype=torch.bool)
    gt_matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return pred_matched, gt_matched

    ious = box_iou(pred_boxes, gt_boxes)
    same_class = pred_labels.unsqueeze(1) == gt_labels.unsqueeze(0)
    ious = ious.masked_fill(~same_class, 0.0)
    for prediction in torch.argsort(pred_scores, descending=True).tolist():
        candidates = ious[prediction].clone()
        candidates[gt_matched] = 0.0
        best_iou, best_gt = candidates.max(dim=0)
        if float(best_iou) >= iou_threshold:
            pred_matched[prediction] = True
            gt_matched[int(best_gt)] = True
    return pred_matched, gt_matched


def f1_score(pred_matched: torch.Tensor, gt_matched: torch.Tensor) -> float:
    true_positives = int(pred_matched.sum())
    predictions = pred_matched.numel()
    ground_truths = gt_matched.numel()
    if predictions == 0 and ground_truths == 0:
        return 1.0
    if true_positives == 0:
        return 0.0
    precision = true_positives / predictions
    recall = true_positives / ground_truths
    return 2 * precision * recall / (precision + recall)


def denormalise(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(OPENAI_CLIP_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(OPENAI_CLIP_STD, dtype=torch.float32).view(3, 1, 1)
    return (image.float() * std + mean).clamp(0.0, 1.0)


def draw_box(
    ax,
    box,
    *,
    status: str,
    label: str,
    linewidth: float,
    font_size: float,
    label_below: bool = False,
):
    """Draw one box with a dark halo so it stays readable over any photograph."""
    import matplotlib.patheffects as path_effects
    from matplotlib.patches import Rectangle

    style = STATUS_STYLES[status]
    x1, y1, x2, y2 = (float(value) for value in box)
    rectangle = Rectangle(
        (x1, y1),
        max(0.0, x2 - x1),
        max(0.0, y2 - y1),
        fill=False,
        linewidth=linewidth,
        edgecolor=style["color"],
        linestyle=style["linestyle"],
    )
    rectangle.set_path_effects(
        [path_effects.withStroke(linewidth=linewidth + 2.0, foreground=HALO, alpha=0.55)]
    )
    ax.add_patch(rectangle)
    if label:
        # Ground truth labels hang below their box and predictions sit above, so
        # the two never stack on top of each other for a well-localised match.
        ax.text(
            x1 + 1,
            y2 + 2 if label_below else max(2.0, y1 - 2),
            label,
            fontsize=font_size,
            color=INK,
            va="top" if label_below else "bottom",
            bbox=dict(
                facecolor=style["color"],
                edgecolor=HALO,
                linewidth=0.4,
                alpha=0.92,
                pad=1.4,
            ),
        )


def draw_panel(
    ax,
    prediction: dict,
    class_names: list[str],
    *,
    title: str,
    match_iou: float,
    max_boxes: int,
    font_size: float,
    show_matched_gt: bool,
    content_size: tuple[float, float] | None = None,
) -> dict:
    image = denormalise(prediction["image"]).permute(1, 2, 0).numpy()
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if content_size is not None:
        # SquarePad pads right and bottom with gray; crop that back off so the
        # panel shows the photo rather than the letterbox.
        ax.set_xlim(0, content_size[0])
        ax.set_ylim(content_size[1], 0)

    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]
    if scores.numel() > max_boxes:
        kept = torch.topk(scores, max_boxes).indices
        boxes, scores, labels = boxes[kept], scores[kept], labels[kept]

    pred_matched, gt_matched = match_predictions(
        boxes, scores, labels, prediction["gt_boxes"], prediction["gt_labels"], match_iou
    )

    for box, is_matched, label in zip(
        prediction["gt_boxes"], gt_matched.tolist(), prediction["gt_labels"].tolist()
    ):
        if is_matched and not show_matched_gt:
            continue
        status = "matched_gt" if is_matched else "missed"
        glyph = STATUS_STYLES[status]["glyph"]
        draw_box(
            ax,
            box,
            status=status,
            label="" if is_matched else f"{glyph} {class_names[label]}".strip(),
            linewidth=1.6 if is_matched else 2.2,
            font_size=font_size,
            label_below=True,
        )

    for box, score, label, is_matched in zip(
        boxes, scores.tolist(), labels.tolist(), pred_matched.tolist()
    ):
        status = "true_positive" if is_matched else "false_positive"
        draw_box(
            ax,
            box,
            status=status,
            label=f"{STATUS_STYLES[status]['glyph']} {class_names[label]} {score:.2f}",
            linewidth=2.0,
            font_size=font_size,
        )

    true_positives = int(pred_matched.sum())
    ax.set_title(
        f"{title}\n{true_positives}/{scores.numel()} predictions matched · "
        f"{int(gt_matched.sum())}/{gt_matched.numel()} ground truth found",
        fontsize=9,
        color=INK,
        pad=8,
    )
    return {
        "true_positives": true_positives,
        "predictions": int(scores.numel()),
        "ground_truths": int(gt_matched.numel()),
        "f1": f1_score(pred_matched, gt_matched),
    }


def add_legend(figure, *, show_matched_gt: bool, columns: int):
    from matplotlib.lines import Line2D

    keys = ["true_positive", "false_positive", "missed"]
    if show_matched_gt:
        keys.append("matched_gt")
    handles = [
        Line2D(
            [0],
            [0],
            color=STATUS_STYLES[key]["color"],
            linestyle=STATUS_STYLES[key]["linestyle"],
            linewidth=2.0,
            label=STATUS_LABELS[key],
        )
        for key in keys
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles) if columns > 1 else 2,
        frameon=False,
        fontsize=8,
        labelcolor=INK_SECONDARY,
        bbox_to_anchor=(0.5, 0.0),
    )


def render_panels(
    detector: TextQueryDetector,
    zero_shot_detector: TextQueryDetector | None,
    val_dataset: CocoDetection,
    indices: list[int],
    class_names: list[str],
    device: torch.device,
    args,
    output_dir: Path,
) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    loader = make_loader(val_dataset, indices, args, batch_size=args.panel_batch_size)
    predict = partial(
        predict_batches,
        loader=loader,
        device=device,
        score_threshold=args.score_threshold,
        top_k=args.max_boxes,
        nms_iou=None if args.nms_iou <= 0 else args.nms_iou,
        score_with_objectness=args.score_with_objectness,
        amp=args.amp,
        keep_images=True,
    )
    tuned = list(predict(detector, description="Panel inference"))
    baseline = (
        list(predict(zero_shot_detector, description="Zero-shot inference"))
        if zero_shot_detector is not None
        else [None] * len(tuned)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for position, (prediction, zero_shot) in enumerate(zip(tuned, baseline)):
        image_id = val_dataset.ids[indices[position]]
        meta = val_dataset.coco.imgs[image_id]
        padded_side = max(float(meta["width"]), float(meta["height"]))
        image_side = float(prediction["image"].shape[-1])
        content_size = (
            float(meta["width"]) / padded_side * image_side,
            float(meta["height"]) / padded_side * image_side,
        )
        columns = 2 if zero_shot is not None else 1
        figure, axes = plt.subplots(
            1, columns, figsize=(7.5 * columns, 8.0), facecolor=SURFACE
        )
        axes = axes if columns > 1 else [axes]
        if zero_shot is not None:
            zero_shot["image"] = prediction["image"]
            draw_panel(
                axes[0],
                zero_shot,
                class_names,
                title="zero-shot OWLv2",
                match_iou=args.match_iou,
                max_boxes=args.max_boxes,
                font_size=args.font_size,
                show_matched_gt=args.show_matched_gt,
                content_size=content_size,
            )
        summary = draw_panel(
            axes[-1],
            prediction,
            class_names,
            title=f"checkpoint · image {image_id}",
            match_iou=args.match_iou,
            max_boxes=args.max_boxes,
            font_size=args.font_size,
            show_matched_gt=args.show_matched_gt,
            content_size=content_size,
        )
        summary["image_id"] = int(image_id)
        summaries.append(summary)

        add_legend(figure, show_matched_gt=args.show_matched_gt, columns=columns)
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        figure.savefig(
            output_dir / f"{position:03d}_image{image_id}.jpg",
            dpi=args.dpi,
            facecolor=SURFACE,
        )
        plt.close(figure)
    return summaries


def make_loader(val_dataset, indices, args, *, batch_size: int) -> DataLoader:
    return DataLoader(
        Subset(val_dataset, indices),
        batch_size=batch_size,
        collate_fn=partial(
            coco_collate_fn,
            id2size=val_dataset.coco.imgs,
            square_pad=True,
            category_id_to_label=args._category_id_to_label,
        ),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args._device.type == "cuda",
    )


def select_indices(
    detector: TextQueryDetector,
    val_dataset: CocoDetection,
    device: torch.device,
    args,
) -> list[int]:
    """Choose which validation images to draw."""
    if args.image_ids:
        id_to_index = {image_id: index for index, image_id in enumerate(val_dataset.ids)}
        missing = [i for i in args.image_ids if i not in id_to_index]
        if missing:
            raise ValueError(f"Validation split has no image ids {missing}")
        return [id_to_index[image_id] for image_id in args.image_ids]

    all_indices = list(range(len(val_dataset)))
    wanted = min(args.num_images, len(all_indices))
    if args.select == "first":
        return all_indices[:wanted]
    if args.select == "random":
        return sorted(random.Random(args.seed).sample(all_indices, wanted))

    # worst/best need a scoring pass first.
    pool = all_indices[: args.scan_limit] if args.scan_limit else all_indices
    loader = make_loader(val_dataset, pool, args, batch_size=args.batch_size)
    scores = []
    for position, prediction in enumerate(
        predict_batches(
            detector,
            loader,
            device,
            score_threshold=args.score_threshold,
            top_k=args.max_boxes,
            nms_iou=None if args.nms_iou <= 0 else args.nms_iou,
            score_with_objectness=args.score_with_objectness,
            amp=args.amp,
            keep_images=False,
            description=f"Ranking {len(pool)} images by F1",
        )
    ):
        pred_matched, gt_matched = match_predictions(
            prediction["boxes"],
            prediction["scores"],
            prediction["labels"],
            prediction["gt_boxes"],
            prediction["gt_labels"],
            args.match_iou,
        )
        # Images with no annotations cannot be ranked meaningfully - an empty
        # prediction set scores a perfect 1.0 and would fill the "best" panels.
        if gt_matched.numel() == 0:
            continue
        scores.append((f1_score(pred_matched, gt_matched), pool[position]))

    scores.sort(key=lambda item: item[0], reverse=args.select == "best")
    return [index for _, index in scores[: args.num_images]]


def compute_metrics(
    detector: TextQueryDetector,
    val_dataset: CocoDetection,
    device: torch.device,
    args,
) -> dict:
    """Full-split metrics under the training-time evaluation protocol."""
    indices = list(range(len(val_dataset)))
    if args.metrics_max_images:
        indices = indices[: args.metrics_max_images]
    loader = make_loader(val_dataset, indices, args, batch_size=args.batch_size)

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=None,
        rec_thresholds=None,
        max_detection_thresholds=[1, 10, 100],
        class_metrics=True,
        backend="faster_coco_eval",
    )
    metric.warn_on_many_detections = False
    for prediction in predict_batches(
        detector,
        loader,
        device,
        score_threshold=args.confidence_threshold,
        top_k=args.eval_top_k,
        nms_iou=None,  # train_text.py evaluates without NMS; keep it comparable.
        score_with_objectness=args.score_with_objectness,
        amp=args.amp,
        keep_images=False,
        description=f"Evaluating {len(indices)} images",
    ):
        metric.update(
            [
                {
                    "boxes": prediction["boxes"],
                    "scores": prediction["scores"],
                    "labels": prediction["labels"],
                }
            ],
            [{"boxes": prediction["gt_boxes"], "labels": prediction["gt_labels"]}],
        )
    return metric.compute()


def per_class_table(
    metrics: dict, class_names: list[str]
) -> tuple[list[tuple[str, float]], list[str]]:
    """Split per-class AP into scored classes and classes with no ground truth.

    torchmetrics reports -1 for a class that never appears in the evaluated
    images, which is not an AP of -1 and must not be plotted as a bar.
    """
    average_precisions = metrics.get("map_per_class")
    classes = metrics.get("classes")
    if average_precisions is None or classes is None or average_precisions.ndim == 0:
        return [], []
    rows, absent = [], []
    for label, value in zip(classes.tolist(), average_precisions.tolist()):
        name = class_names[label] if 0 <= label < len(class_names) else f"class {label}"
        if value < 0:
            absent.append(name)
        else:
            rows.append((name, float(value)))
    return sorted(rows, key=lambda row: row[1]), sorted(absent)


def plot_per_class_ap(rows: list[tuple[str, float]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    names = [name for name, _ in rows]
    values = [value for _, value in rows]
    figure, ax = plt.subplots(
        figsize=(8.0, max(4.0, 0.24 * len(rows) + 1.5)), facecolor=SURFACE
    )
    ax.set_facecolor(SURFACE)
    positions = range(len(rows))
    ax.barh(list(positions), values, height=0.72, color=BAR_COLOR)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(names, fontsize=6.5, color=INK_SECONDARY)
    ax.tick_params(axis="x", labelsize=7, colors=INK_SECONDARY, length=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("AP @ IoU 0.50:0.95", fontsize=8, color=INK_SECONDARY)
    ax.set_title("Per-class AP", fontsize=10, color=INK, loc="left", pad=10)
    ax.xaxis.grid(True, color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#d9d8d4")
    ax.set_ylim(-0.8, len(rows) - 0.2)

    # Label only the extremes: a number on all 46 bars is noise, and the full
    # table is printed to stdout and written to metrics.txt anyway.
    highlighted = set(list(positions)[:5] + list(positions)[-5:])
    for position in highlighted:
        ax.text(
            values[position] + 0.004,
            position,
            f"{values[position]:.3f}",
            va="center",
            fontsize=6.5,
            color=INK_SECONDARY,
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def format_metrics(
    metrics: dict, rows: list[tuple[str, float]], absent: list[str]
) -> str:
    lines = ["=== Overall ==="]
    for key in (
        "map",
        "map_50",
        "map_75",
        "map_small",
        "map_medium",
        "map_large",
        "mar_1",
        "mar_10",
        "mar_100",
    ):
        value = metrics.get(key)
        if value is not None and getattr(value, "numel", lambda: 1)() == 1:
            lines.append(f"{key:<12} {float(value):.4f}")
    if rows:
        width = max(len(name) for name, _ in rows)
        lines.append("\n=== Per-class AP (worst first) ===")
        lines.extend(f"{name:<{width}}  {value:.4f}" for name, value in rows)
    if absent:
        lines.append(
            f"\n{len(absent)} classes absent from the evaluated images (no AP): "
            + ", ".join(absent)
        )
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Path to a .pth, a run directory (best.pth is preferred inside it), "
            "or a bare run name under --checkpoint-root."
        ),
    )
    parser.add_argument("--checkpoint-root", default="text_checkpoints")
    parser.add_argument(
        "--val-annotations",
        help="Defaults to the val_annotations recorded in the checkpoint config.",
    )
    parser.add_argument(
        "--val-images",
        help="Defaults to the val_images recorded in the checkpoint config.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for the metric and ranking passes.")
    parser.add_argument("--panel-batch-size", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--val-transform",
        choices=("fast", "accurate"),
        help="Defaults to the transform the run was evaluated with.",
    )
    parser.add_argument(
        "--prompt-template",
        help=(
            "Re-derive the evaluation prompts from this template instead of using "
            "the ones stored in the checkpoint. Must contain {name}."
        ),
    )
    parser.add_argument("--output-dir", default="eval_visualizations")

    parser.add_argument(
        "--select",
        choices=("worst", "best", "random", "first"),
        default="worst",
        help="How to choose panel images. worst/best rank by per-image F1.",
    )
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument(
        "--image-ids",
        nargs="+",
        type=int,
        help="Draw these COCO image ids instead of using --select.",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=200,
        help="Images scored when ranking for --select worst/best. 0 scans the split.",
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.1,
        help="Panel-only score floor. The metric pass uses --confidence-threshold.",
    )
    parser.add_argument("--max-boxes", type=int, default=20, help="Predictions drawn per panel.")
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="Panel-only class-wise NMS. 0 disables it. Never applied to the metric pass.",
    )
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument(
        "--show-matched-gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also outline ground truth that was found, to judge localisation quality.",
    )
    parser.add_argument("--font-size", type=float, default=6.5)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument(
        "--compare-zero-shot",
        action="store_true",
        help="Add a stock-OWLv2 panel beside each checkpoint panel.",
    )

    parser.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--metrics-max-images",
        type=int,
        help="Cap the metric pass. Omit to evaluate the whole split.",
    )
    parser.add_argument(
        "--panels", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--eval-top-k", type=int)
    parser.add_argument("--score-with-objectness", action=argparse.BooleanOptionalAction)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    if args.num_images <= 0:
        raise ValueError("num_images must be positive")

    checkpoint_path = resolve_checkpoint_path(args.checkpoint, Path(args.checkpoint_root))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["_path"] = checkpoint_path
    config = checkpoint.get("config", {})
    print(
        f"Checkpoint {checkpoint_path} ({checkpoint.get('format')}), "
        f"epoch {checkpoint.get('epoch')}, recorded map {checkpoint.get('map')}"
    )

    # Anything the caller did not override comes from the run that produced the
    # checkpoint, so the defaults reproduce that run's evaluation exactly.
    args.val_annotations = args.val_annotations or config_default(config, "val_annotations")
    args.val_images = args.val_images or config_default(config, "val_images")
    if not args.val_annotations or not args.val_images:
        raise ValueError(
            "The checkpoint has no recorded validation paths; pass "
            "--val-annotations and --val-images"
        )
    args.val_transform = args.val_transform or config_default(config, "val_transform", "fast")
    if args.confidence_threshold is None:
        args.confidence_threshold = config_default(config, "confidence_threshold", 0.001)
    if args.eval_top_k is None:
        args.eval_top_k = config_default(config, "eval_top_k", 100)
    if args.score_with_objectness is None:
        args.score_with_objectness = config_default(config, "score_with_objectness", True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    args._device = device
    args.amp = args.amp and device.type == "cuda"

    class_names = list(checkpoint["class_names"])
    detector, prompts = build_detector(checkpoint, class_names, args.prompt_template, device)
    print(f"{len(class_names)} classes, first prompts: {prompts[:3]}")

    val_dataset = CocoDetection(
        annFile=args.val_annotations,
        root=args.val_images,
        transform=(
            detector.owl.image_transform_fast
            if args.val_transform == "fast"
            else detector.owl.image_transform_accurate
        ),
    )
    category_ids = config_default(config, "category_ids") or sorted(
        val_dataset.coco.getCatIds()
    )
    if len(category_ids) != len(class_names):
        raise ValueError(
            f"Checkpoint has {len(class_names)} classes but the validation split "
            f"exposes {len(category_ids)} category ids"
        )
    val_names = coco_class_names(val_dataset.coco, category_ids)
    if val_names != class_names:
        differing = [
            f"{index}: {left!r} != {right!r}"
            for index, (left, right) in enumerate(zip(class_names, val_names))
            if left != right
        ]
        raise ValueError(f"Class names disagree with the validation split: {differing[:5]}")
    args._category_id_to_label = {
        category_id: label for label, category_id in enumerate(category_ids)
    }

    output_dir = Path(args.output_dir) / checkpoint_path.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.metrics:
        metrics = compute_metrics(detector, val_dataset, device, args)
        rows, absent = per_class_table(metrics, class_names)
        report = format_metrics(metrics, rows, absent)
        print(f"\n{report}")
        (output_dir / "metrics.txt").write_text(
            f"checkpoint: {checkpoint_path}\n"
            f"prompts: {prompts[0] if prompts else ''} ...\n"
            f"protocol: threshold {args.confidence_threshold}, top-k {args.eval_top_k}, "
            f"objectness {args.score_with_objectness}, no NMS\n\n{report}\n"
        )
        if rows:
            plot_per_class_ap(rows, output_dir / "per_class_ap.png")
            print(f"Wrote {output_dir / 'per_class_ap.png'}")

    if args.panels:
        indices = select_indices(detector, val_dataset, device, args)
        if not indices:
            raise RuntimeError("No validation images were selected")
        zero_shot_detector = (
            build_zero_shot_detector(checkpoint["model_type"], prompts, device)
            if args.compare_zero_shot
            else None
        )
        summaries = render_panels(
            detector,
            zero_shot_detector,
            val_dataset,
            indices,
            class_names,
            device,
            args,
            output_dir / "panels",
        )
        (output_dir / "panels.json").write_text(json.dumps(summaries, indent=2))
        print(
            f"\nWrote {len(summaries)} panels to {output_dir / 'panels'} "
            f"(--select {args.select}, score >= {args.score_threshold})"
        )
        for summary in summaries:
            print(
                f"  image {summary['image_id']}: F1 {summary['f1']:.2f}, "
                f"{summary['true_positives']}/{summary['predictions']} predictions matched, "
                f"{summary['ground_truths']} ground truth boxes"
            )


if __name__ == "__main__":
    main()
