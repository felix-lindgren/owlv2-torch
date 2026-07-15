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
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OWLv2torch.torch_version.owlv2 import OwlV2
from OWLv2torch.torch_version.text_loss import compute_text_query_losses
from OWLv2torch.utils.tokenizer import tokenize
from prototype_train.train import (
    AugmentedDetectionDataset,
    aug_collate_fn,
    coco_class_names,
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
) -> None:
    """Save a small trainable-parameter delta unless ``--save-full-model`` is set."""
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
        },
        path,
    )


def load_text_checkpoint(model: OwlV2, checkpoint_path: str | Path) -> dict:
    """Load either a full or delta checkpoint produced by this trainer."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=checkpoint.get("format") == "owlv2-text-finetune-full-v1",
    )
    if checkpoint.get("format") != "owlv2-text-finetune-full-v1" and incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
    return checkpoint


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-annotations", required=True)
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--val-annotations", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--model-type", choices=("base", "large"), default="base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--category-ids", nargs="+", type=int)
    parser.add_argument("--shots-per-class", type=int)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
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
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

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
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--max-negatives-per-image", type=int, default=512)
    parser.add_argument("--negative-loss-weight", type=float, default=1.0)
    parser.add_argument("--lambda-class", type=float, default=1.0)
    parser.add_argument("--lambda-l1", type=float, default=5.0)
    parser.add_argument("--lambda-giou", type=float, default=2.0)
    parser.add_argument("--lambda-objectness", type=float, default=0.5)

    parser.add_argument("--confidence-threshold", type=float, default=0.001)
    parser.add_argument("--eval-top-k", type=int, default=300)
    parser.add_argument("--score-with-objectness", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-zero-shot-baseline", action="store_true")
    parser.add_argument("--output-dir", default="text_checkpoints")
    parser.add_argument("--save-full-model", action="store_true")
    parser.add_argument("--mlflow", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mlflow-experiment", default="OwlV2-Text-Training")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.eval_every < 0:
        raise ValueError("eval_every must be non-negative")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.eval_every_steps is not None and args.eval_every_steps <= 0:
        raise ValueError("eval_every_steps must be positive")
    if args.eval_max_batches is not None and args.eval_max_batches <= 0:
        raise ValueError("eval_max_batches must be positive")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    prompt_generator = random.Random(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    if args.mlflow:
        configure_mlflow_tracking()
        mlflow.set_experiment(args.mlflow_experiment)
        run_context = mlflow.start_run(
            run_name=f"owlv2_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    else:
        run_context = nullcontext()

    with run_context:
        model = OwlV2(args.model_type)
        parameter_groups = configure_trainable_parameter_groups(model, args)
        if not parameter_groups:
            raise RuntimeError("No trainable parameter groups were configured")

        train_dataset = CocoDetection(
            annFile=args.train_annotations,
            root=args.train_images,
        )
        val_dataset = CocoDetection(
            annFile=args.val_annotations,
            root=args.val_images,
            transform=model.image_transform,
        )
        category_ids = args.category_ids or sorted(train_dataset.coco.getCatIds())
        missing_val_ids = set(category_ids) - set(val_dataset.coco.getCatIds())
        if missing_val_ids:
            raise ValueError(f"Validation annotations are missing category ids {sorted(missing_val_ids)}")
        category_id_to_label = {
            category_id: label for label, category_id in enumerate(category_ids)
        }
        class_names = coco_class_names(train_dataset.coco, category_ids)
        prompt_templates = args.prompt_templates or list(DEFAULT_PROMPT_TEMPLATES)
        prompt_pools = build_prompt_pools(class_names, prompt_templates)
        eval_prompts = evaluation_prompts(class_names, args.eval_prompt_template)
        detector = TextQueryDetector(model, eval_prompts).to(device)

        train_indices = select_shot_indices(
            train_dataset,
            category_ids,
            args.shots_per_class,
            args.seed,
        )
        train_augmented = AugmentedDetectionDataset(
            train_dataset,
            model.image_transform_fast,
            category_id_to_label,
            augment=args.augment,
            random_right_angle_rotation=False,
            horizontal_flip_prob=0.5,
            vertical_flip_prob=0.0,
        )
        train_loader = DataLoader(
            Subset(train_augmented, train_indices),
            batch_size=args.batch_size,
            collate_fn=aug_collate_fn,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_steps or args.epochs
        )
        amp_enabled = args.amp and device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

        trainable_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        print(f"Classes: {len(class_names)}")
        print(f"Training images: {len(train_indices)}")
        print(f"Trainable parameters: {trainable_count:,}")
        for group in parameter_groups:
            count = sum(parameter.numel() for parameter in group["params"])
            print(f"  {group['name']}: {count:,} parameters at lr={group['lr']:.2g}")

        if args.mlflow:
            mlflow.log_params(
                {
                    **{key: str(value) for key, value in vars(args).items()},
                    "num_classes": len(class_names),
                    "class_names": str(class_names),
                    "train_images_selected": len(train_indices),
                    "trainable_parameters": trainable_count,
                }
            )

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
                for key, value in baseline_metrics.items():
                    mlflow.log_metric(f"baseline/{key}", mlflow_metric_value(value))

        output_dir = Path(args.output_dir)
        best_map = float("-inf")
        latest_map = None
        global_step = 0
        last_eval_step = None
        completed_epoch = 0
        if len(train_loader) == 0:
            raise RuntimeError("The training loader is empty")
        if args.max_steps is None:
            epochs_to_run = args.epochs
        else:
            epochs_to_run = (args.max_steps + len(train_loader) - 1) // len(train_loader)

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
                for key, value in metrics.items():
                    mlflow.log_metric(
                        f"val/{key}", mlflow_metric_value(value), step=step
                    )
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

        for epoch in range(epochs_to_run):
            detector.train()
            running_loss = 0.0
            epoch_steps = 0
            progress_total = len(train_loader)
            if args.max_steps is not None:
                progress_total = min(progress_total, args.max_steps - global_step)
            progress = tqdm(
                train_loader,
                desc=f"Text training epoch {epoch + 1}",
                total=progress_total,
            )
            for batch in progress:
                if args.max_steps is not None and global_step >= args.max_steps:
                    break
                images = batch["images"].to(device, non_blocking=True)
                prompts = sample_prompt_set(prompt_pools, prompt_generator)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    outputs = detector(images, prompts=prompts)
                    losses = compute_text_query_losses(
                        outputs,
                        batch["targets"],
                        lambda_cls=args.lambda_class,
                        lambda_l1=args.lambda_l1,
                        lambda_giou=args.lambda_giou,
                        lambda_objectness=args.lambda_objectness,
                        negative_ratio=args.negative_ratio,
                        max_negatives_per_image=args.max_negatives_per_image,
                        negative_loss_weight=args.negative_loss_weight,
                    )
                scaler.scale(losses["loss"]).backward()
                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        (parameter for parameter in model.parameters() if parameter.requires_grad),
                        args.grad_clip_norm,
                    )
                scaler.step(optimizer)
                scaler.update()

                current_loss = float(losses["loss"].detach())
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
                    mlflow.log_metric("train/loss_step", current_loss, step=global_step)

                if (
                    args.eval_every_steps is not None
                    and global_step % args.eval_every_steps == 0
                ):
                    run_validation(epoch + 1, global_step)
                    detector.train()

                if args.max_steps is not None and global_step >= args.max_steps:
                    break

            completed_epoch = epoch + 1
            if args.max_steps is None:
                scheduler.step()
            average_loss = running_loss / max(1, epoch_steps)
            print(f"Epoch {epoch + 1}: average loss {average_loss:.4f}")
            if args.mlflow:
                mlflow.log_metric("train/loss_epoch", average_loss, step=epoch + 1)
                for group in optimizer.param_groups:
                    mlflow.log_metric(
                        f"train/lr/{group['name']}", group["lr"], step=epoch + 1
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
