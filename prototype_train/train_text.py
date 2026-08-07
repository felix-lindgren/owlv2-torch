"""Fine-tune OWLv2 as a text-conditioned detector on a COCO-style dataset.

Unlike ``prototype_train/train.py``, this trainer passes natural-language
queries through the text tower on every step. The resulting checkpoint remains
usable with arbitrary text queries at inference time.
"""

from __future__ import annotations

import argparse
import random
import sys
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from pathlib import Path

import mlflow
import torch
from mlflow.exceptions import MlflowException
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OWLv2torch.torch_version.owlv2 import OwlV2
from OWLv2torch.torch_version.text_loss import (
    CLASS_LOSS_CHOICES,
    compute_text_query_losses,
)
from OWLv2torch.utils.tokenizer import tokenize
from prototype_train.gpu_augment import BatchAugmentor
from prototype_train.train import (
    AugmentedDetectionDataset,
    aug_collate_fn,
    coco_collate_fn,
    coco_eval,
    configure_mlflow_tracking,
    mlflow_metric_value,
    select_shot_indices,
)


DEFAULT_PROMPT_TEMPLATES = (
    "{name}",
    "a photo of {name}",
    "a person wearing {name}",
)


def _clean_query_text(text: str) -> str:
    return " ".join(text.strip().lower().replace("-", " ").replace("_", " ").split())


def build_prompt_pools(
    class_names: list[str],
    templates: list[str] | tuple[str, ...] = DEFAULT_PROMPT_TEMPLATES,
) -> list[list[str]]:
    """Build prompt variants, treating comma-separated category names as aliases."""
    if not templates:
        raise ValueError("At least one prompt template is required")
    pools = []
    invalid_templates = [template for template in templates if "{name}" not in template]
    if invalid_templates:
        raise ValueError(
            f"Prompt templates must contain a '{{name}}' placeholder: {invalid_templates}"
        )
    for class_name in class_names:
        full_name = _clean_query_text(class_name)
        aliases = [_clean_query_text(alias) for alias in class_name.split(",")]
        names = list(dict.fromkeys([full_name, *filter(None, aliases)]))
        prompts = []
        for template in templates:
            try:
                variants = [_clean_query_text(template.format(name=name)) for name in names]
            except (KeyError, IndexError) as exc:
                raise ValueError(
                    f"Prompt template {template!r} must use a '{{name}}' placeholder"
                ) from exc
            prompts.extend(variant for variant in variants if variant)
        prompts = list(dict.fromkeys(prompts))
        if not prompts:
            raise ValueError(f"No prompts could be built for class {class_name!r}")
        pools.append(prompts)
    return pools


def drop_categories_by_name(
    coco, category_ids: list[int], excluded: list[str]
) -> tuple[list[int], list[str]]:
    """Remove categories whose name matches ``excluded``, order otherwise preserved.

    Matching is on the cleaned full category name, so a multi-alias category such
    as ``"shirt, blouse"`` must be named in full. An unmatched entry raises rather
    than silently keeping the class it was meant to drop.
    """
    wanted = {_clean_query_text(name) for name in excluded}
    cleaned_by_id = {
        category_id: _clean_query_text(coco.cats[category_id]["name"])
        for category_id in category_ids
    }
    unmatched = sorted(wanted - set(cleaned_by_id.values()))
    if unmatched:
        available = ", ".join(sorted(cleaned_by_id.values()))
        raise ValueError(
            f"--exclude-class-names did not match any category: {unmatched}. "
            f"Available: {available}"
        )
    kept = [
        category_id
        for category_id in category_ids
        if cleaned_by_id[category_id] not in wanted
    ]
    if not kept:
        raise ValueError("--exclude-class-names removed every category")
    dropped = [
        cleaned_by_id[category_id]
        for category_id in category_ids
        if cleaned_by_id[category_id] in wanted
    ]
    return kept, dropped


def build_label_space(
    coco, category_ids: list[int], merge_specs: list[str]
) -> tuple[list[str], dict[int, int], list[tuple[list[str], str]]]:
    """Map category ids onto query indices, optionally merging groups of categories.

    Without ``merge_specs`` this is the identity: one query per category, in the
    order of ``category_ids``. Each spec is ``source+source[+...]=target``, whose
    sources are matched against cleaned category names exactly as
    ``--exclude-class-names`` does. ``+`` separates sources rather than ``,``
    because a comma is already the alias separator inside a single category name
    (Fashionpedia's ``"shirt, blouse"``).

    Merging maps several category ids onto one query index. Unlike dropping a
    category, every box stays supervised -- which is the whole point for a lateral
    pair, where dropping one side would leave visually identical objects labelled
    on one side and unlabelled on the other.

    The merged class takes the position of its first source, and a target naming
    a category that is being kept merges into that category rather than creating
    a second query with the same text.
    """
    cleaned_by_id = {
        category_id: _clean_query_text(coco.cats[category_id]["name"])
        for category_id in category_ids
    }
    available = set(cleaned_by_id.values())

    target_by_source: dict[str, str] = {}
    merges: list[tuple[list[str], str]] = []
    for spec in merge_specs:
        sources_text, separator, target = spec.partition("=")
        target = target.strip()
        if not separator or not target:
            raise ValueError(
                f"--merge-class-names entry {spec!r} is not of the form "
                "'source+source=target'"
            )
        sources = [
            cleaned
            for cleaned in (_clean_query_text(part) for part in sources_text.split("+"))
            if cleaned
        ]
        if len(sources) < 2:
            raise ValueError(
                f"--merge-class-names entry {spec!r} needs at least two source "
                "classes separated by '+'"
            )
        for source in sources:
            if source not in available:
                raise ValueError(
                    f"--merge-class-names source {source!r} did not match any "
                    f"category. Available: {', '.join(sorted(available))}"
                )
            if source in target_by_source:
                raise ValueError(
                    f"--merge-class-names maps {source!r} into both "
                    f"{target_by_source[source]!r} and {target!r}"
                )
            target_by_source[source] = target
        merges.append((sources, target))

    class_names: list[str] = []
    label_by_name: dict[str, int] = {}
    category_id_to_label: dict[int, int] = {}
    for category_id in category_ids:
        name = target_by_source.get(
            cleaned_by_id[category_id], coco.cats[category_id]["name"]
        )
        key = _clean_query_text(name)
        if key not in label_by_name:
            label_by_name[key] = len(class_names)
            class_names.append(name)
        category_id_to_label[category_id] = label_by_name[key]
    return class_names, category_id_to_label, merges


def sample_prompt_set(
    prompt_pools: list[list[str]], generator: random.Random
) -> list[str]:
    """Choose one independently sampled phrase for each class/query index."""
    return [generator.choice(pool) for pool in prompt_pools]


class TextQueryDetector(nn.Module):
    """OWLv2 wrapper with fixed evaluation prompts and variable training prompts."""

    def __init__(self, owl: OwlV2, evaluation_prompts: list[str]):
        super().__init__()
        if not evaluation_prompts:
            raise ValueError("At least one evaluation prompt is required")
        self.owl = owl
        self.num_queries = len(evaluation_prompts)
        token_ids = tokenize(evaluation_prompts, context_length=16, truncate=True)
        self.register_buffer("evaluation_token_ids", token_ids, persistent=False)

    def forward(self, pixel_values: torch.Tensor, prompts: list[str] | None = None):
        if prompts is None:
            token_ids = self.evaluation_token_ids
        else:
            if len(prompts) != self.num_queries:
                raise ValueError(
                    f"Expected {self.num_queries} prompts, got {len(prompts)}"
                )
            token_ids = tokenize(prompts, context_length=16, truncate=True).to(
                pixel_values.device
            )
        attention_mask = token_ids == 0
        return self.owl.forward_object_detection(pixel_values, token_ids, attention_mask)


def configure_trainable_parameter_groups(model: OwlV2, args) -> list[dict]:
    """Freeze the base model, then enable the requested domain-adaptation modules."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    groups = []

    def add_group(name: str, module: nn.Module, learning_rate: float):
        parameters = list(module.parameters())
        for parameter in parameters:
            parameter.requires_grad_(True)
        if parameters:
            groups.append({"params": parameters, "lr": learning_rate, "name": name})

    add_group("class_head", model.class_head, args.head_learning_rate)
    if args.train_box_head:
        add_group("box_head", model.box_head, args.head_learning_rate)
    if args.train_objectness_head:
        add_group("objectness_head", model.objectness_head, args.head_learning_rate)

    vision_layers = model.vision_model.encoder.layers
    if not 0 <= args.vision_blocks <= len(vision_layers):
        raise ValueError(
            f"vision_blocks must be in [0, {len(vision_layers)}], got {args.vision_blocks}"
        )
    if args.vision_blocks:
        add_group(
            "vision_blocks",
            nn.ModuleList(vision_layers[-args.vision_blocks :]),
            args.vision_learning_rate,
        )
        add_group("vision_post_norm", model.vision_model.post_layernorm, args.vision_learning_rate)
        add_group("detection_layer_norm", model.layer_norm, args.vision_learning_rate)

    text_layers = model.text_model.encoder.layers
    if not 0 <= args.text_blocks <= len(text_layers):
        raise ValueError(
            f"text_blocks must be in [0, {len(text_layers)}], got {args.text_blocks}"
        )
    if args.text_blocks:
        add_group(
            "text_blocks",
            nn.ModuleList(text_layers[-args.text_blocks :]),
            args.text_learning_rate,
        )
        add_group("text_final_norm", model.text_model.final_layer_norm, args.text_learning_rate)
        add_group("text_projection", model.text_projection, args.text_learning_rate)

    return groups


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics, treating a failing tracking store as non-fatal.

    The SQLite store raises ``database is locked`` when a concurrent run holds the
    write lock. The per-step call site runs thousands of times, so letting that
    propagate takes down a multi-hour run to save a single metric point.
    """
    try:
        mlflow.log_metrics(metrics, step=step)
    except MlflowException as error:
        print(f"Skipped MLflow metrics at step {step}: {error}", file=sys.stderr)


def log_eval_metrics(metrics, prefix: str, step: int | None = None) -> None:
    """Log the scalar entries of a torchmetrics result, skipping per-class tensors."""
    scalars = {
        f"{prefix}{key}": mlflow_metric_value(value)
        for key, value in metrics.items()
        if not (hasattr(value, "numel") and value.numel() != 1)
    }
    log_metrics(scalars, step=step)


def evaluation_prompts(class_names: list[str], template: str) -> list[str]:
    if "{name}" not in template:
        raise ValueError("The evaluation prompt template must contain a '{name}' placeholder")
    return [_clean_query_text(template.format(name=_clean_query_text(name))) for name in class_names]


def save_checkpoint(
    path: Path,
    model: OwlV2,
    class_names: list[str],
    eval_prompts: list[str],
    args,
    *,
    epoch: int,
    map_value: float | None,
    training_state: dict | None = None,
) -> None:
    """Save a small trainable-parameter delta unless ``--save-full-model`` is set.

    ``training_state`` holds everything ``--resume`` needs beyond the weights.
    Only ``last.pth`` carries it: the AdamW moments alone are twice the size of
    the trainable delta, and nothing but ``--resume`` ever reads them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.save_full_model:
        model_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        checkpoint_format = "owlv2-text-finetune-full-v1"
    else:
        trainable_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        model_state = {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
            if name in trainable_names
        }
        checkpoint_format = "owlv2-text-finetune-delta-v1"
    torch.save(
        {
            "format": checkpoint_format,
            "model_type": args.model_type,
            "model_state_dict": model_state,
            "class_names": class_names,
            "evaluation_prompts": eval_prompts,
            "epoch": epoch,
            "map": map_value,
            "config": vars(args),
            "training_state": training_state,
        },
        path,
    )


def apply_text_checkpoint(model: OwlV2, checkpoint: dict) -> dict:
    """Load an already-read checkpoint's weights into ``model``."""
    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=checkpoint.get("format") == "owlv2-text-finetune-full-v1",
    )
    if checkpoint.get("format") != "owlv2-text-finetune-full-v1" and incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
    return checkpoint


def load_text_checkpoint(model: OwlV2, checkpoint_path: str | Path) -> dict:
    """Load either a full or delta checkpoint produced by this trainer."""
    return apply_text_checkpoint(
        model, torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    )


# Changing any of these makes a resumed run neither a continuation of the old one
# nor a clean new one, and the failure is silent: the optimizer state is keyed by
# parameter position, the scheduler by step count, and the learning rates live in
# the restored optimizer state, so a changed --vision-learning-rate is ignored
# rather than applied.
RESUME_CRITICAL_ARGS = (
    "model_type",
    "vision_blocks",
    "text_blocks",
    "train_box_head",
    "train_objectness_head",
    "head_learning_rate",
    "vision_learning_rate",
    "text_learning_rate",
    "weight_decay",
    "batch_size",
    "grad_accum_steps",
    "max_steps",
    "epochs",
    "warmup_steps",
    "seed",
    "shots_per_class",
    "amp",
)


def check_resume_compatibility(
    checkpoint: dict, args, class_names: list[str] | None = None
) -> None:
    """Reject a resume whose configuration differs from the run being continued.

    The class set is only compared when ``class_names`` is supplied, so the
    argument comparison can run before the tracking run is opened: a rejected
    resume should not mark the run it was going to continue as failed.
    """
    saved = checkpoint.get("config") or {}
    mismatches = [
        f"{name}: checkpoint {saved.get(name)!r}, requested {getattr(args, name)!r}"
        for name in RESUME_CRITICAL_ARGS
        if name in saved and saved[name] != getattr(args, name)
    ]
    saved_classes = checkpoint.get("class_names")
    if (
        class_names is not None
        and saved_classes is not None
        and list(saved_classes) != list(class_names)
    ):
        mismatches.append(
            f"class set: checkpoint has {len(saved_classes)} classes, "
            f"this run has {len(class_names)}"
        )
    if mismatches:
        raise ValueError(
            "--resume checkpoint does not match this run's configuration:\n  "
            + "\n  ".join(mismatches)
            + "\nUse --init-from to start a new run from these weights instead."
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-annotations", required=True)
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--val-annotations", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--model-type", choices=("base", "large"), default="base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--category-ids", nargs="+", type=int)
    parser.add_argument(
        "--exclude-class-names",
        nargs="+",
        help=(
            "Drop these COCO categories, matched against category names after the "
            "same cleaning applied to prompts (an unmatched name is an error). "
            "Their boxes leave both the training targets and the validation "
            "ground truth, and their queries leave the prompt set, so the metric "
            "is computed over the remaining classes only and is NOT comparable to "
            "a run with a different class set."
        ),
    )
    parser.add_argument(
        "--merge-class-names",
        nargs="+",
        metavar="SOURCE+SOURCE=TARGET",
        help=(
            "Collapse groups of COCO categories into one query, e.g. "
            "'left_arm+right_arm=arm'. Sources are matched like "
            "--exclude-class-names; '+' separates them because a comma already "
            "separates aliases within one category name. Every box stays "
            "supervised under the merged label, unlike --exclude-class-names, "
            "which would leave one side of a lateral pair unlabelled. This is a "
            "metric change: mAP over the merged set is NOT comparable to a run "
            "with a different class set."
        ),
    )
    parser.add_argument("--shots-per-class", type=int)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--val-batch-size",
        type=int,
        help="Validation batch size. Defaults to --batch-size when omitted.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Stop after this many optimizer steps, regardless of epoch boundaries.",
    )
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        help="Evaluate every N optimizer steps instead of using --eval-every.",
    )
    parser.add_argument(
        "--eval-max-batches",
        type=int,
        help="Use at most this many validation batches per evaluation.",
    )
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gpu-augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run mosaic, colour jitter and normalisation on the device with "
            "kornia instead of jittering per sample in the dataloader workers. "
            "Required by --mosaic-prob. Needs --augment."
        ),
    )
    parser.add_argument(
        "--mosaic-prob",
        type=float,
        default=0.5,
        help=(
            "Per-image probability of replacing the image with a mosaic of "
            "--mosaic-grid images drawn from the same batch. 0 disables it."
        ),
    )
    parser.add_argument(
        "--mosaic-grid",
        type=int,
        nargs=2,
        default=(2, 2),
        metavar=("ROWS", "COLS"),
        help="Mosaic tiling. The output keeps the model's input resolution.",
    )
    parser.add_argument(
        "--mosaic-start-ratio",
        type=float,
        nargs=2,
        default=(0.3, 0.7),
        metavar=("LOW", "HIGH"),
        help=(
            "Range the mosaic crop's top-left corner is sampled from, as a "
            "fraction of the input size. Values near 0.5 keep all tiles visible."
        ),
    )
    parser.add_argument(
        "--mosaic-min-visibility",
        type=float,
        default=0.2,
        help=(
            "Drop a mosaic box when the mosaic crop leaves less than this "
            "fraction of its original area."
        ),
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the vision encoder for faster training (about 1.10x end-to-end).",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help=(
            "Micro-batches accumulated per optimizer step, for an effective batch "
            "of --batch-size x --grad-accum-steps. --max-steps, --eval-every-steps "
            "and the scheduler all count optimizer steps, so keep samples seen "
            "constant by dividing --max-steps by this value. Note this is not "
            "numerically identical to the same effective batch in one go: each "
            "micro-batch's loss is normalised by its own box/image counts and the "
            "micro-batch losses are then averaged, so a sparse image carries the "
            "same weight as a crowded one."
        ),
    )
    parser.add_argument(
        "--grad-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Recompute encoder-block activations during backward instead of "
            "storing them. Roughly halves activation memory (allowing larger "
            "--batch-size) at about a 20%% step-time cost. Blocks that are frozen and fed a "
            "detached input are skipped, so this is a no-op with --vision-blocks 0 "
            "--text-blocks 0."
        ),
    )
    parser.add_argument(
        "--val-transform",
        choices=("fast", "accurate"),
        default="fast",
        help=(
            "Validation preprocessing. 'fast' matches the transform used for training "
            "and is far cheaper on CPU; 'accurate' uses the scipy resize."
        ),
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help=(
            "Linearly warm up the learning rates over this many scheduler steps. "
            "Scheduler steps are optimizer steps with --max-steps, epochs otherwise."
        ),
    )

    parser.add_argument("--head-learning-rate", type=float, default=1e-5)
    parser.add_argument("--vision-learning-rate", type=float, default=1e-6)
    parser.add_argument("--text-learning-rate", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--vision-blocks", type=int, default=0)
    parser.add_argument("--text-blocks", type=int, default=0)
    parser.add_argument(
        "--train-box-head", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--train-objectness-head", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument(
        "--prompt-template",
        action="append",
        dest="prompt_templates",
        help="Repeat for prompt augmentation; must contain {name}",
    )
    parser.add_argument("--eval-prompt-template", default="a photo of {name}")
    parser.add_argument(
        "--class-loss",
        choices=CLASS_LOSS_CHOICES,
        default="focal",
        help=(
            "Classification term. 'focal' is the matched + hard-negative-mined "
            "split; 'mal' (DEIM) and 'vfl' are dense IoU-aware losses that ignore "
            "the negative-mining flags. Box and objectness terms are identical "
            "across all three, so runs are directly comparable."
        ),
    )
    parser.add_argument(
        "--class-loss-gamma",
        type=float,
        default=1.5,
        help="Focusing exponent for --class-loss mal/vfl (DEIM uses 1.5).",
    )
    parser.add_argument(
        "--class-loss-alpha",
        type=float,
        help=(
            "Background weight for --class-loss mal/vfl. Default: unscaled for "
            "mal, 0.2 for vfl, matching DEIM."
        ),
    )
    parser.add_argument(
        "--negative-ratio",
        type=int,
        default=5,
        help="Negatives mined per positive. Used by --class-loss focal and by the objectness term.",
    )
    parser.add_argument("--max-negatives-per-image", type=int, default=512)
    parser.add_argument("--negative-loss-weight", type=float, default=1.0)
    parser.add_argument("--lambda-class", type=float, default=1.0)
    parser.add_argument("--lambda-l1", type=float, default=5.0)
    parser.add_argument("--lambda-giou", type=float, default=2.0)
    parser.add_argument("--lambda-objectness", type=float, default=0.5)

    parser.add_argument("--confidence-threshold", type=float, default=0.001)
    parser.add_argument(
        "--eval-top-k",
        type=int,
        default=100,
        help=(
            "Detections kept per image for evaluation. torchmetrics caps AP at 100 "
            "detections, so larger values only cost CPU time in the metric update."
        ),
    )
    parser.add_argument("--score-with-objectness", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-zero-shot-baseline", action="store_true")
    parser.add_argument(
        "--init-from",
        help=(
            "Load model weights from a checkpoint written by this trainer, then "
            "train from scratch optimizer-wise. The checkpoint's class set need "
            "not match this run's. Use --resume to continue an interrupted run."
        ),
    )
    parser.add_argument(
        "--resume",
        help=(
            "Continue an interrupted run from its last.pth (or a directory "
            "containing one). Restores weights, optimizer, scheduler, AMP "
            "scaler, step counters, best mAP and RNG state, and fast-forwards "
            "the epoch's sample order to where it stopped. The configuration "
            "must match the run being continued."
        ),
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Run one validation pass and exit without training or writing "
            "checkpoints. With --init-from and --exclude-class-names this scores "
            "existing weights against a different query set."
        ),
    )
    parser.add_argument("--output-dir", default="text_checkpoints")
    parser.add_argument("--save-full-model", action="store_true")
    parser.add_argument("--mlflow", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mlflow-experiment", default="OwlV2-Text-Training")
    parser.add_argument(
        "--mlflow-run-name",
        help="Run name in MLflow. Defaults to a timestamp.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.val_batch_size is not None and args.val_batch_size <= 0:
        raise ValueError("val_batch_size must be positive")
    if args.eval_every < 0:
        raise ValueError("eval_every must be non-negative")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.eval_every_steps is not None and args.eval_every_steps <= 0:
        raise ValueError("eval_every_steps must be positive")
    if args.eval_max_batches is not None and args.eval_max_batches <= 0:
        raise ValueError("eval_max_batches must be positive")
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if args.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be at least 1")
    if not 0.0 <= args.mosaic_prob <= 1.0:
        raise ValueError("mosaic_prob must be in [0, 1]")
    gpu_augment = args.augment and args.gpu_augment
    if args.mosaic_prob > 0 and not gpu_augment:
        raise ValueError("--mosaic-prob requires --augment and --gpu-augment")
    if args.resume and args.init_from:
        raise ValueError("--resume and --init-from are mutually exclusive")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    prompt_generator = random.Random(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    # Read before the tracking run is opened so a resume can rejoin the run it
    # was interrupted in rather than starting a second one.
    resume_checkpoint = None
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            resume_path = resume_path / "last.pth"
        resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if not resume_checkpoint.get("training_state"):
            raise ValueError(
                f"{resume_path} carries no training state, so it cannot be "
                "resumed: it either predates --resume support or is a "
                "best.pth/final.pth. Only last.pth stores optimizer state."
            )
        check_resume_compatibility(resume_checkpoint, args)

    if args.mlflow:
        configure_mlflow_tracking()
        mlflow.set_experiment(args.mlflow_experiment)
        resumed_run_id = (
            resume_checkpoint["training_state"].get("mlflow_run_id")
            if resume_checkpoint is not None
            else None
        )
        if resumed_run_id:
            # Same run, so val/map stays a single series across the interruption.
            # Train metrics between the last checkpoint and the crash are logged
            # twice at those steps, once per attempt.
            run_context = mlflow.start_run(run_id=resumed_run_id)
        else:
            run_context = mlflow.start_run(
                run_name=(
                    args.mlflow_run_name
                    or f"owlv2_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
            )
    else:
        run_context = nullcontext()

    with run_context:
        model = OwlV2(args.model_type)
        parameter_groups = configure_trainable_parameter_groups(model, args)
        if not parameter_groups:
            raise RuntimeError("No trainable parameter groups were configured")
        if args.grad_checkpointing:
            model.set_gradient_checkpointing(True)

        train_dataset = CocoDetection(
            annFile=args.train_annotations,
            root=args.train_images,
        )
        val_dataset = CocoDetection(
            annFile=args.val_annotations,
            root=args.val_images,
            transform=(
                model.image_transform_fast
                if args.val_transform == "fast"
                else model.image_transform_accurate
            ),
        )
        category_ids = args.category_ids or sorted(train_dataset.coco.getCatIds())
        if args.exclude_class_names:
            category_ids, excluded_names = drop_categories_by_name(
                train_dataset.coco, category_ids, args.exclude_class_names
            )
            print(f"Excluded {len(excluded_names)} classes: {', '.join(excluded_names)}")
        missing_val_ids = set(category_ids) - set(val_dataset.coco.getCatIds())
        if missing_val_ids:
            raise ValueError(f"Validation annotations are missing category ids {sorted(missing_val_ids)}")
        class_names, category_id_to_label, merges = build_label_space(
            train_dataset.coco, category_ids, args.merge_class_names or []
        )
        for sources, target in merges:
            print(f"Merged {' + '.join(sources)} -> {target}")

        if resume_checkpoint is not None:
            check_resume_compatibility(resume_checkpoint, args, class_names)
            apply_text_checkpoint(model, resume_checkpoint)
            print(f"Resuming from {resume_path}")
        elif args.init_from:
            initial_checkpoint = load_text_checkpoint(model, args.init_from)
            print(f"Initialised weights from {args.init_from}")
            initial_classes = initial_checkpoint.get("class_names") or []
            if list(initial_classes) != list(class_names):
                print(
                    f"  checkpoint was trained on {len(initial_classes)} classes; "
                    f"this run uses {len(class_names)}"
                )

        prompt_templates = args.prompt_templates or list(DEFAULT_PROMPT_TEMPLATES)
        prompt_pools = build_prompt_pools(class_names, prompt_templates)
        eval_prompts = evaluation_prompts(class_names, args.eval_prompt_template)
        detector = TextQueryDetector(model, eval_prompts).to(device)
        if args.compile:
            model.vision_model.encoder.compile()

        train_indices = select_shot_indices(
            train_dataset,
            category_ids,
            args.shots_per_class,
            args.seed,
        )
        # With GPU augmentation the workers stop at the unnormalised square
        # image; mosaic and the colour jitter want [0, 1] inputs, so the
        # normalisation moves to the end of the device-side pipeline.
        train_augmented = AugmentedDetectionDataset(
            train_dataset,
            model.image_transform_unnormed if gpu_augment else model.image_transform_fast,
            category_id_to_label,
            augment=args.augment,
            random_right_angle_rotation=False,
            horizontal_flip_prob=0.5,
            vertical_flip_prob=0.0,
            photometric=not gpu_augment,
        )
        batch_augmentor = (
            BatchAugmentor(
                model.image_size,
                mosaic_prob=args.mosaic_prob,
                mosaic_grid=tuple(args.mosaic_grid),
                mosaic_start_ratio_range=tuple(args.mosaic_start_ratio),
                min_box_visibility=args.mosaic_min_visibility,
            ).to(device)
            if gpu_augment
            else None
        )
        def make_train_loader(epoch: int, skip_batches: int = 0) -> DataLoader:
            """Build one epoch's loader over an explicit, resumable sample order.

            ``shuffle=True`` draws from the DataLoader's own RNG, which cannot be
            positioned partway through an epoch. Materialising the permutation
            here makes ``skip_batches`` an exact fast-forward — the resumed epoch
            sees precisely the samples the interrupted one had left, and in the
            same order — at the cost of the order differing from a pre-resume
            run of the same seed.

            The generator is passed to the loader as well, so worker
            augmentation seeds depend only on the epoch rather than on where the
            global RNG happens to be. Worker RNG is still not restored exactly on
            resume: a worker re-seeded at the start of a truncated epoch reaches
            a given sample at a different point in its stream.
            """
            generator = torch.Generator().manual_seed(args.seed * 1_000_003 + epoch)
            order = torch.randperm(len(train_indices), generator=generator).tolist()
            epoch_indices = [
                train_indices[position]
                for position in order[skip_batches * args.batch_size :]
            ]
            return DataLoader(
                Subset(train_augmented, epoch_indices),
                batch_size=args.batch_size,
                collate_fn=aug_collate_fn,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                prefetch_factor=4 if args.num_workers > 0 else None,
                drop_last=True,
                generator=generator,
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size or args.batch_size,
            collate_fn=partial(
                coco_collate_fn,
                id2size=val_dataset.coco.imgs,
                square_pad=True,
                category_id_to_label=category_id_to_label,
            ),
            shuffle=False,
            num_workers=max(0, args.num_workers // 2),
            pin_memory=device.type == "cuda",
        )

        optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=args.weight_decay,
        )
        scheduler_steps = args.max_steps or args.epochs
        if args.warmup_steps:
            if args.warmup_steps >= scheduler_steps:
                raise ValueError(
                    f"warmup_steps must be smaller than the {scheduler_steps} scheduler "
                    "steps in the run"
                )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer, start_factor=0.1, total_iters=args.warmup_steps
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=scheduler_steps - args.warmup_steps
                    ),
                ],
                milestones=[args.warmup_steps],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=scheduler_steps
            )
        amp_enabled = args.amp and device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

        trainable_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        print(f"Classes: {len(class_names)}")
        print(f"Training images: {len(train_indices)}")
        print(
            f"Effective batch: {args.batch_size * args.grad_accum_steps} "
            f"({args.batch_size} x {args.grad_accum_steps} accumulated)"
        )
        print(f"Trainable parameters: {trainable_count:,}")
        for group in parameter_groups:
            count = sum(parameter.numel() for parameter in group["params"])
            print(f"  {group['name']}: {count:,} parameters at lr={group['lr']:.2g}")

        if args.mlflow and resume_checkpoint is None:
            mlflow.log_params(
                {
                    **{key: str(value) for key, value in vars(args).items()},
                    "num_classes": len(class_names),
                    "class_names": str(class_names),
                    "train_images_selected": len(train_indices),
                    "trainable_parameters": trainable_count,
                    "effective_batch_size": args.batch_size * args.grad_accum_steps,
                }
            )
        elif args.mlflow:
            # The run already carries its parameters, and MLflow rejects
            # re-logging one with a different value -- which --resume itself is.
            mlflow.set_tag("resumed_from", str(resume_path))

        if args.run_zero_shot_baseline:
            baseline_metrics = coco_eval(
                detector,
                val_loader,
                device,
                confidence_threshold=args.confidence_threshold,
                score_with_objectness=args.score_with_objectness,
                top_k=args.eval_top_k,
                output_format="standard",
                title="Text-conditioned baseline",
                log_to_mlflow=args.mlflow,
                max_batches=args.eval_max_batches,
            )
            if args.mlflow:
                log_eval_metrics(baseline_metrics, "baseline/")

        if args.eval_only:
            metrics = coco_eval(
                detector,
                val_loader,
                device,
                confidence_threshold=args.confidence_threshold,
                score_with_objectness=args.score_with_objectness,
                top_k=args.eval_top_k,
                output_format="standard",
                title=f"Text-conditioned evaluation ({len(class_names)} classes)",
                log_to_mlflow=args.mlflow,
                max_batches=args.eval_max_batches,
            )
            if args.mlflow:
                log_eval_metrics(metrics, "val/", step=0)
            print(f"Evaluation mAP over {len(class_names)} classes: {float(metrics['map']):.4f}")
            return

        output_dir = Path(args.output_dir)
        best_map = float("-inf")
        latest_map = None
        global_step = 0
        last_eval_step = None
        completed_epoch = 0
        start_epoch = 0
        resume_batches_done = 0
        accum_steps = args.grad_accum_steps
        full_epoch_batches = len(train_indices) // args.batch_size
        if full_epoch_batches == 0:
            raise RuntimeError("The training loader is empty")
        steps_per_epoch = (full_epoch_batches + accum_steps - 1) // accum_steps

        if resume_checkpoint is not None:
            state = resume_checkpoint["training_state"]
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            if state["scaler"]:
                scaler.load_state_dict(state["scaler"])
            global_step = state["global_step"]
            start_epoch = state["epoch_index"]
            resume_batches_done = state["epoch_batches_done"]
            completed_epoch = start_epoch
            best_map = state["best_map"]
            latest_map = state["latest_map"]
            last_eval_step = state["last_eval_step"]
            torch.set_rng_state(state["torch_rng_state"])
            if state["cuda_rng_state"] is not None and device.type == "cuda":
                torch.cuda.set_rng_state_all(state["cuda_rng_state"])
            random.setstate(state["python_rng_state"])
            prompt_generator.setstate(state["prompt_rng_state"])
            print(
                f"Resumed at optimizer step {global_step}, epoch {start_epoch + 1}, "
                f"{resume_batches_done}/{full_epoch_batches} batches into that epoch "
                f"(best mAP so far {best_map:.4f})"
            )
            if args.max_steps is not None and global_step >= args.max_steps:
                raise ValueError(
                    f"Checkpoint is already at step {global_step} of "
                    f"--max-steps {args.max_steps}; nothing left to run"
                )

        if args.max_steps is None:
            epochs_to_run = args.epochs
        else:
            # The resumed epoch is short, so it gets counted separately from the
            # full epochs that follow it.
            first_epoch_batches = full_epoch_batches - resume_batches_done
            first_epoch_steps = (first_epoch_batches + accum_steps - 1) // accum_steps
            remaining_steps = args.max_steps - global_step
            if remaining_steps <= first_epoch_steps:
                epochs_to_run = start_epoch + 1
            else:
                epochs_to_run = start_epoch + 1 + (
                    remaining_steps - first_epoch_steps + steps_per_epoch - 1
                ) // steps_per_epoch

        # Position within the current epoch's permutation, so an interrupted run
        # can pick the sample order back up where it stopped.
        current_epoch_index = start_epoch
        current_epoch_batches_done = resume_batches_done

        def build_training_state() -> dict:
            return {
                "global_step": global_step,
                "epoch_index": current_epoch_index,
                "epoch_batches_done": current_epoch_batches_done,
                "best_map": best_map,
                "latest_map": latest_map,
                "last_eval_step": last_eval_step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                ),
                "python_rng_state": random.getstate(),
                "prompt_rng_state": prompt_generator.getstate(),
                "mlflow_run_id": (
                    mlflow.active_run().info.run_id if args.mlflow else None
                ),
            }

        def run_validation(epoch_number: int, step: int) -> None:
            nonlocal best_map, latest_map, last_eval_step
            metrics = coco_eval(
                detector,
                val_loader,
                device,
                confidence_threshold=args.confidence_threshold,
                score_with_objectness=args.score_with_objectness,
                top_k=args.eval_top_k,
                output_format="standard",
                title=(
                    f"Text-conditioned validation epoch {epoch_number}, step {step}"
                ),
                log_to_mlflow=args.mlflow,
                max_batches=args.eval_max_batches,
            )
            latest_map = float(metrics["map"])
            last_eval_step = step
            if args.mlflow:
                log_eval_metrics(metrics, "val/", step=step)
            if latest_map > best_map:
                best_map = latest_map
                save_checkpoint(
                    output_dir / "best.pth",
                    model,
                    class_names,
                    eval_prompts,
                    args,
                    epoch=epoch_number,
                    map_value=latest_map,
                )
                print(f"Saved new best checkpoint with mAP {best_map:.4f}")
            # Written after best.pth so the resume state carries the updated
            # best_map, and a resumed run does not re-save an inferior best.
            save_checkpoint(
                output_dir / "last.pth",
                model,
                class_names,
                eval_prompts,
                args,
                epoch=epoch_number,
                map_value=latest_map,
                training_state=build_training_state(),
            )

        for epoch in range(start_epoch, epochs_to_run):
            batch_offset = resume_batches_done if epoch == start_epoch else 0
            train_loader = make_train_loader(epoch, batch_offset)
            num_batches = len(train_loader)
            current_epoch_index = epoch
            current_epoch_batches_done = batch_offset
            detector.train()
            running_loss = 0.0
            epoch_steps = 0
            progress_total = num_batches
            if args.max_steps is not None:
                progress_total = min(
                    progress_total, (args.max_steps - global_step) * accum_steps
                )
            progress = tqdm(
                train_loader,
                desc=f"Text training epoch {epoch + 1}",
                total=progress_total,
            )
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            window_size = 0
            window_losses: dict[str, float] = {}
            for batch_index, batch in enumerate(progress):
                current_epoch_batches_done = batch_offset + batch_index + 1
                if accumulated == 0:
                    # The last window of an epoch can be short, so normalise by the
                    # number of micro-batches it actually holds rather than by
                    # accum_steps: a truncated window keeps the same effective lr.
                    window_size = min(accum_steps, num_batches - batch_index)
                    window_losses = {}
                images = batch["images"].to(device, non_blocking=True)
                targets = batch["targets"]
                if batch_augmentor is not None:
                    images, targets = batch_augmentor(images, targets)
                prompts = sample_prompt_set(prompt_pools, prompt_generator)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    outputs = detector(images, prompts=prompts)
                    losses = compute_text_query_losses(
                        outputs,
                        targets,
                        lambda_cls=args.lambda_class,
                        lambda_l1=args.lambda_l1,
                        lambda_giou=args.lambda_giou,
                        lambda_objectness=args.lambda_objectness,
                        negative_ratio=args.negative_ratio,
                        max_negatives_per_image=args.max_negatives_per_image,
                        negative_loss_weight=args.negative_loss_weight,
                        class_loss=args.class_loss,
                        class_loss_gamma=args.class_loss_gamma,
                        class_loss_alpha=args.class_loss_alpha,
                    )
                scaler.scale(losses["loss"] / window_size).backward()
                for name, value in losses.items():
                    window_losses[name] = (
                        window_losses.get(name, 0.0) + float(value.detach()) / window_size
                    )
                accumulated += 1
                if accumulated < window_size:
                    continue
                accumulated = 0

                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        (parameter for parameter in model.parameters() if parameter.requires_grad),
                        args.grad_clip_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                current_loss = window_losses["loss"]
                running_loss += current_loss
                global_step += 1
                epoch_steps += 1
                if args.max_steps is not None:
                    scheduler.step()
                progress.set_postfix(
                    loss=f"{current_loss:.4f}",
                    step=(f"{global_step}/{args.max_steps}" if args.max_steps else global_step),
                )
                if args.mlflow:
                    log_metrics(
                        {
                            "train/loss_step": current_loss,
                            **{
                                f"train/{name}": value
                                for name, value in window_losses.items()
                                if name != "loss"
                            },
                        },
                        step=global_step,
                    )

                if (
                    args.eval_every_steps is not None
                    and global_step % args.eval_every_steps == 0
                ):
                    run_validation(epoch + 1, global_step)
                    detector.train()

                if args.max_steps is not None and global_step >= args.max_steps:
                    break

            if not (args.max_steps is not None and global_step >= args.max_steps):
                # The epoch was consumed rather than cut short, so a checkpoint
                # written below should resume at the start of the next one.
                current_epoch_index = epoch + 1
                current_epoch_batches_done = 0
            completed_epoch = epoch + 1
            if args.max_steps is None:
                scheduler.step()
            average_loss = running_loss / max(1, epoch_steps)
            print(f"Epoch {epoch + 1}: average loss {average_loss:.4f}")
            if args.mlflow:
                log_metrics(
                    {
                        "train/loss_epoch": average_loss,
                        **{
                            f"train/lr/{group['name']}": group["lr"]
                            for group in optimizer.param_groups
                        },
                    },
                    step=epoch + 1,
                )

            should_evaluate_by_epoch = (
                args.eval_every_steps is None
                and (
                    (args.eval_every > 0 and (epoch + 1) % args.eval_every == 0)
                    or epoch + 1 == epochs_to_run
                )
            )
            if should_evaluate_by_epoch and last_eval_step != global_step:
                run_validation(epoch + 1, global_step)

            if args.max_steps is not None and global_step >= args.max_steps:
                break

        if last_eval_step != global_step:
            run_validation(completed_epoch, global_step)

        save_checkpoint(
            output_dir / "final.pth",
            model,
            class_names,
            eval_prompts,
            args,
            epoch=completed_epoch,
            map_value=latest_map,
        )
        print(f"Saved final checkpoint to {output_dir / 'final.pth'}")
        if best_map != float("-inf"):
            print(f"Best validation mAP: {best_map:.4f}")


if __name__ == "__main__":
    main()
