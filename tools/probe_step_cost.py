"""Probe training step time and peak GPU memory across a config grid (run-plan L0a).

Replaces the missing ``tools/probe_grad_ckpt_memory.py``. Answers three questions
before a training budget is committed: how long a step costs, which
``--batch-size`` still fits in the card, and whether ``--compile`` and
``--grad-checkpointing`` are worth their cost on this model.

The measured step reproduces ``prototype_train/train_text.py``'s inner loop --
pinned host-to-device image copy, GPU augmentation, prompt sampling and
tokenization, autocast forward, the loss (including Hungarian matching), scaled
backward, grad clipping and the optimizer step -- so the number is comparable to
wall clock per step. Only the dataloader is excluded: its cost overlaps with GPU
work in the real trainer and would otherwise be measured twice.

Box density drives the Hungarian match, so targets are read from a real
annotations file rather than invented. Images are random tensors: the vision
tower's cost does not depend on pixel content.

Each grid cell runs in its own subprocess. An out-of-memory cell is then a
recorded result rather than a poisoned CUDA context that corrupts the cells
after it.

    uv run --extra train python tools/probe_step_cost.py \
      --annotations /mnt/datasets/fashion/lv_mhp_coco/train/annotations.json \
      --device cuda:0 --model-type base \
      --vision-blocks 0 2 6 --batch-sizes 8 16 32 --compile-modes off on
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OWLv2torch.torch_version.owlv2 import OwlV2
from OWLv2torch.torch_version.text_loss import compute_text_query_losses
from prototype_train.gpu_augment import BatchAugmentor
from prototype_train.train_text import (
    DEFAULT_PROMPT_TEMPLATES,
    TextQueryDetector,
    build_prompt_pools,
    configure_trainable_parameter_groups,
    evaluation_prompts,
    sample_prompt_set,
)

RESULT_PREFIX = "PROBE_RESULT "
GIB = 1024 ** 3


def load_target_pool(
    annotations_path: str, max_images: int
) -> tuple[list[dict[str, torch.Tensor]], list[str]]:
    """Read per-image targets in the trainer's format: normalised cxcywh, square-padded.

    The trainer pads to a square of ``max(width, height)`` before resizing, so a
    box normalises by that side on both axes.
    """
    with open(annotations_path) as handle:
        data = json.load(handle)
    sizes = {image["id"]: (image["width"], image["height"]) for image in data["images"]}
    categories = sorted(data["categories"], key=lambda category: category["id"])
    label_of = {category["id"]: label for label, category in enumerate(categories)}
    class_names = [category["name"] for category in categories]

    by_image: dict[int, list[dict]] = {}
    for annotation in data["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)

    pool = []
    for image_id, annotations in list(by_image.items())[:max_images]:
        width, height = sizes[image_id]
        side = float(max(width, height))
        boxes = torch.tensor(
            [
                [
                    (annotation["bbox"][0] + annotation["bbox"][2] * 0.5) / side,
                    (annotation["bbox"][1] + annotation["bbox"][3] * 0.5) / side,
                    annotation["bbox"][2] / side,
                    annotation["bbox"][3] / side,
                ]
                for annotation in annotations
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [label_of[annotation["category_id"]] for annotation in annotations],
            dtype=torch.int64,
        )
        keep = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
        pool.append({"boxes": boxes[keep], "labels": labels[keep]})
    if not pool:
        raise ValueError(f"No annotated images found in {annotations_path}")
    return pool, class_names


def synthetic_target_pool(
    boxes_per_image: float, num_classes: int, num_images: int, seed: int
) -> tuple[list[dict[str, torch.Tensor]], list[str]]:
    """Stand-in used when no annotations file is given."""
    generator = torch.Generator().manual_seed(seed)
    pool = []
    for _ in range(num_images):
        count = max(1, int(round(boxes_per_image)))
        centres = torch.rand((count, 2), generator=generator) * 0.8 + 0.1
        sizes = torch.rand((count, 2), generator=generator) * 0.15 + 0.05
        pool.append(
            {
                "boxes": torch.cat([centres, sizes], dim=1),
                "labels": torch.randint(0, num_classes, (count,), generator=generator),
            }
        )
    return pool, [f"class {index}" for index in range(num_classes)]


def run_cell(args) -> dict:
    """Measure one (vision_blocks, batch_size, compile, checkpointing) configuration."""
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    generator = random.Random(args.seed)

    if args.annotations:
        target_pool, class_names = load_target_pool(args.annotations, args.target_images)
    else:
        target_pool, class_names = synthetic_target_pool(
            args.boxes_per_image, args.num_classes, 256, args.seed
        )
    boxes_per_image = statistics.mean(
        float(target["boxes"].shape[0]) for target in target_pool
    )

    model = OwlV2(args.model_type)
    trainer_args = Namespace(
        head_learning_rate=5e-5,
        vision_learning_rate=1e-5,
        text_learning_rate=1e-7,
        train_box_head=True,
        train_objectness_head=True,
        vision_blocks=args.vision_blocks,
        text_blocks=0,
    )
    parameter_groups = configure_trainable_parameter_groups(model, trainer_args)
    if args.grad_checkpointing:
        model.set_gradient_checkpointing(True)

    prompt_pools = build_prompt_pools(class_names, list(DEFAULT_PROMPT_TEMPLATES))
    detector = TextQueryDetector(
        model, evaluation_prompts(class_names, "a photo of {name}")
    ).to(device)
    if args.compile:
        model.vision_model.encoder.compile()
    detector.train()

    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-4)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    augmentor = (
        BatchAugmentor(
            model.image_size,
            mosaic_prob=args.mosaic_prob,
            mosaic_grid=tuple(args.mosaic_grid),
        ).to(device)
        if args.gpu_augment
        else None
    )

    # The trainer's loader hands over pinned, unnormalised [0, 1] square images;
    # the copy is part of the step, so it is measured rather than hoisted.
    host_images = torch.rand(
        args.batch_size, 3, model.image_size, model.image_size
    ).pin_memory()
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    def one_step(step_index: int) -> None:
        offset = (step_index * args.batch_size) % len(target_pool)
        targets = [
            target_pool[(offset + index) % len(target_pool)]
            for index in range(args.batch_size)
        ]
        images = host_images.to(device, non_blocking=True)
        if augmentor is not None:
            images, targets = augmentor(images, targets)
        prompts = sample_prompt_set(prompt_pools, generator)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = detector(images, prompts=prompts)
            losses = compute_text_query_losses(
                outputs,
                targets,
                lambda_cls=1.0,
                lambda_l1=5.0,
                lambda_giou=2.0,
                lambda_objectness=0.5,
                class_loss=args.class_loss,
                class_loss_gamma=1.5,
            )
        scaler.scale(losses["loss"]).backward()
        for value in losses.values():
            float(value.detach())  # the per-step sync the trainer already does
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    result = {
        "model_type": args.model_type,
        "vision_blocks": args.vision_blocks,
        "batch_size": args.batch_size,
        "compile": bool(args.compile),
        "grad_checkpointing": bool(args.grad_checkpointing),
        "amp": bool(amp_enabled),
        "trainable_parameters": trainable_count,
        "boxes_per_image": round(boxes_per_image, 2),
        "num_classes": len(class_names),
        "status": "ok",
    }

    try:
        warmup_start = time.perf_counter()
        one_step(0)
        torch.cuda.synchronize(device)
        result["first_step_s"] = round(time.perf_counter() - warmup_start, 2)
        for step_index in range(1, args.warmup):
            one_step(step_index)
        torch.cuda.synchronize(device)

        # Reset after warmup so the reported peak is the steady state a long run
        # actually sustains, not the compiler's one-off workspace.
        torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for step_index in range(args.steps):
            start = time.perf_counter()
            one_step(args.warmup + step_index)
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - start) * 1000.0)
    except torch.cuda.OutOfMemoryError as error:
        result["status"] = "oom"
        result["error"] = str(error).splitlines()[0]
        return result

    result["step_ms"] = round(statistics.median(samples), 1)
    result["step_ms_sd"] = round(
        statistics.stdev(samples) if len(samples) > 1 else 0.0, 1
    )
    result["ms_per_sample"] = round(statistics.median(samples) / args.batch_size, 1)
    result["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated(device) / GIB, 2)
    result["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved(device) / GIB, 2)
    result["total_memory_gib"] = round(
        torch.cuda.get_device_properties(device).total_memory / GIB, 2
    )
    return result


def format_table(results: list[dict], epoch_images: int, project_steps: int) -> str:
    header = (
        "| vb | bs | compile | ckpt | step ms | sd | ms/sample | img/s | "
        f"peak alloc | peak resv | 1st step | {epoch_images}-img epoch | "
        f"{project_steps} steps |"
    )
    lines = [header, "|" + "---|" * 13]
    for result in results:
        if result["status"] != "ok":
            lines.append(
                f"| {result['vision_blocks']} | {result['batch_size']} | "
                f"{'on' if result['compile'] else 'off'} | "
                f"{'on' if result['grad_checkpointing'] else 'off'} | "
                f"**{result['status'].upper()}** | | | | | | | | |"
            )
            continue
        step_s = result["step_ms"] / 1000.0
        images_per_second = result["batch_size"] / step_s
        epoch_minutes = epoch_images / images_per_second / 60.0
        project_hours = project_steps * step_s / 3600.0
        lines.append(
            f"| {result['vision_blocks']} | {result['batch_size']} | "
            f"{'on' if result['compile'] else 'off'} | "
            f"{'on' if result['grad_checkpointing'] else 'off'} | "
            f"{result['step_ms']:.0f} | {result['step_ms_sd']:.0f} | "
            f"{result['ms_per_sample']:.0f} | {images_per_second:.1f} | "
            f"{result['peak_allocated_gib']:.2f} GiB | "
            f"{result['peak_reserved_gib']:.2f} GiB | "
            f"{result['first_step_s']:.0f} s | {epoch_minutes:.1f} min | "
            f"{project_hours:.2f} h |"
        )
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=("base", "large"), default="base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--annotations",
        help=(
            "COCO annotations whose per-image box counts and class list the probe "
            "reuses. Falls back to --boxes-per-image synthetic targets if omitted."
        ),
    )
    parser.add_argument(
        "--target-images",
        type=int,
        default=256,
        help="Images read from --annotations to cycle through as target sets.",
    )
    parser.add_argument("--boxes-per-image", type=float, default=25.0)
    parser.add_argument("--num-classes", type=int, default=18)
    parser.add_argument("--vision-blocks", type=int, nargs="+", default=[0, 2, 6])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument(
        "--compile-modes",
        nargs="+",
        choices=("off", "on"),
        default=["off", "on"],
        help="A/B for --compile; 'on' calls vision_model.encoder.compile().",
    )
    parser.add_argument(
        "--grad-ckpt-modes",
        nargs="+",
        choices=("off", "on"),
        default=["off"],
        help="A/B for --grad-checkpointing.",
    )
    parser.add_argument("--class-loss", default="mal")
    parser.add_argument("--mosaic-prob", type=float, default=0.0)
    parser.add_argument("--mosaic-grid", type=int, nargs=2, default=(2, 2))
    parser.add_argument("--gpu-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--epoch-images",
        type=int,
        default=3600,
        help="Training-set size used for the derived epoch-time column.",
    )
    parser.add_argument(
        "--project-steps",
        type=int,
        default=4500,
        help="Step budget used for the derived run-time column.",
    )
    parser.add_argument("--output-json", help="Write the raw per-cell results here.")
    # Set by the parent process; one cell per subprocess.
    parser.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--compile", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--grad-checkpointing", action="store_true", help=argparse.SUPPRESS
    )
    return parser


def cell_command(args, vision_blocks: int, batch_size: int, compiled: bool, ckpt: bool):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cell",
        "--model-type", args.model_type,
        "--device", args.device,
        "--vision-blocks", str(vision_blocks),
        "--batch-size", str(batch_size),
        "--class-loss", args.class_loss,
        "--mosaic-prob", str(args.mosaic_prob),
        "--warmup", str(args.warmup),
        "--steps", str(args.steps),
        "--seed", str(args.seed),
        "--target-images", str(args.target_images),
        "--boxes-per-image", str(args.boxes_per_image),
        "--num-classes", str(args.num_classes),
    ]
    if args.annotations:
        command += ["--annotations", args.annotations]
    command += ["--gpu-augment"] if args.gpu_augment else ["--no-gpu-augment"]
    command += ["--amp"] if args.amp else ["--no-amp"]
    if compiled:
        command.append("--compile")
    if ckpt:
        command.append("--grad-checkpointing")
    return command


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)

    if args.cell:
        if args.batch_size is None:
            raise ValueError("--cell requires --batch-size")
        args.vision_blocks = args.vision_blocks[0] if isinstance(
            args.vision_blocks, list
        ) else args.vision_blocks
        print(RESULT_PREFIX + json.dumps(run_cell(args)))
        return

    cells = [
        (vision_blocks, batch_size, compiled == "on", ckpt == "on")
        for ckpt in args.grad_ckpt_modes
        for compiled in args.compile_modes
        for vision_blocks in args.vision_blocks
        for batch_size in args.batch_sizes
    ]
    print(
        f"{len(cells)} cells on {args.device}: "
        f"vb={args.vision_blocks} x bs={args.batch_sizes} x "
        f"compile={args.compile_modes} x ckpt={args.grad_ckpt_modes}"
    )

    results = []
    started = time.perf_counter()
    for index, (vision_blocks, batch_size, compiled, ckpt) in enumerate(cells, start=1):
        label = (
            f"vb{vision_blocks} bs{batch_size} "
            f"compile={'on' if compiled else 'off'} ckpt={'on' if ckpt else 'off'}"
        )
        print(f"[{index}/{len(cells)}] {label} ...", flush=True)
        completed = subprocess.run(
            cell_command(args, vision_blocks, batch_size, compiled, ckpt),
            capture_output=True,
            text=True,
        )
        payload = None
        for line in completed.stdout.splitlines():
            if line.startswith(RESULT_PREFIX):
                payload = json.loads(line[len(RESULT_PREFIX):])
        if payload is None:
            tail = (completed.stderr or completed.stdout).strip().splitlines()[-3:]
            status = "oom" if "out of memory" in (completed.stderr or "").lower() else "error"
            payload = {
                "model_type": args.model_type,
                "vision_blocks": vision_blocks,
                "batch_size": batch_size,
                "compile": compiled,
                "grad_checkpointing": ckpt,
                "status": status,
                "error": " | ".join(tail),
            }
            print(f"    {status}: {payload['error']}", flush=True)
        elif payload["status"] == "ok":
            print(
                f"    {payload['step_ms']:.0f} ms/step, "
                f"{payload['peak_reserved_gib']:.2f} GiB reserved",
                flush=True,
            )
        else:
            print(f"    {payload['status']}: {payload.get('error', '')}", flush=True)
        results.append(payload)

    ok = [result for result in results if result["status"] == "ok"]
    print(f"\n{len(ok)}/{len(cells)} cells measured in {(time.perf_counter() - started) / 60:.1f} min")
    if ok:
        print(
            f"model={args.model_type}  "
            f"{ok[0]['num_classes']} classes  "
            f"{ok[0]['boxes_per_image']:.1f} boxes/image  "
            f"amp={ok[0]['amp']}  card={ok[0]['total_memory_gib']:.1f} GiB\n"
        )
    print(format_table(results, args.epoch_images, args.project_steps))

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
