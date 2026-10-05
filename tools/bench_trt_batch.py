"""Benchmark the batched TensorRT OWLv2 vision tower at several batch sizes.

For every batch size this times one batch of:

  preprocess  CPU image transform (exact scipy, or ``fast`` torchvision), stacked
  h2d         host-to-device copy of the stacked pixels
  trt tower   the TensorRT vision tower alone
  torch tower the fp16-autocast torch vision tower, for reference
  detect      the full detection forward exactly as ovd_eval runs it
  e2e         preprocess + h2d + detect + sync, i.e. what an eval batch costs

An out-of-memory batch is reported as a row instead of killing the sweep.

Example: ``python tools/bench_trt_batch.py --model-size large --device cuda:0``
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from OWLv2torch import tokenize
from ovd_eval import AERIAL_QUERY_TEMPLATE, build_owlv2_model, detect

QUERY_NAMES = ["car", "truck", "bus", "van", "person", "bicycle", "motorcycle", "boat"]


def gpu_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def cpu_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / iters


def device_used_gib(device) -> float:
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / 2**30


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-size", default="base", choices=["base", "large"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--image", type=Path, default=REPO_ROOT / "img.jpg")
    parser.add_argument("--trt-engine", type=Path, default=None)
    parser.add_argument("--trt-engine-dir", type=Path, default=REPO_ROOT / "trt_engines")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--no-torch-tower", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    baseline_gib = device_used_gib(device)

    model = build_owlv2_model(
        args.model_size,
        str(device),
        tensorrt=True,
        engine_path=args.trt_engine,
        engine_dir=args.trt_engine_dir,
        batch_size=max(args.batch_sizes),
    )
    engine = model.trt.engine
    profile = engine.get_tensor_profile_shape("image", 0)
    trt_scratch_gib = engine.device_memory_size_v2 / 2**30
    print(
        f"model={args.model_size} S={model.image_size} device={device} "
        f"profile min/opt/max batch={profile[0][0]}/{profile[1][0]}/{profile[2][0]} "
        f"trt scratch={trt_scratch_gib:.2f} GiB"
    )
    print(f"loaded: {device_used_gib(device) - baseline_gib:.2f} GiB on device")

    queries = [AERIAL_QUERY_TEMPLATE.format(name=n) for n in QUERY_NAMES]
    token_ids = tokenize(queries, context_length=16, truncate=True).to(device)
    attention_mask = token_ids == 0
    image = Image.open(args.image).convert("RGB")
    print(f"image {args.image.name} {image.size[0]}x{image.size[1]}, {len(queries)} queries\n")

    header = (
        f"{'B':>2} {'prep':>5} | {'preproc':>8} {'h2d':>6} {'trt twr':>8} {'torch twr':>9} "
        f"{'detect':>7} {'e2e':>7} | {'img/s':>6} {'twr img/s':>9} | {'torch pk':>8} {'dev used':>8}"
    )
    print(header)
    print("-" * len(header))

    for batch_size in args.batch_sizes:
        images = [image] * batch_size
        for fast in (False, True):
            prep_name = "fast" if fast else "exact"
            try:
                torch.cuda.reset_peak_memory_stats(device)
                # Exact scipy preprocessing is ~0.3 s/image; fewer iterations keep
                # the sweep short without changing the per-batch mean much.
                prep_iters = args.iters if fast else max(2, args.iters // 3)
                preproc = cpu_ms(
                    lambda: model.preprocess_image(images, fast=fast), 1, prep_iters
                )
                pixels_cpu = model.preprocess_image(images, fast=fast)
                h2d = cpu_ms(lambda: pixels_cpu.to(device), 1, args.iters)
                pixels = pixels_cpu.to(device)

                with torch.inference_mode():
                    trt_tower = gpu_ms(lambda: model.trt(image=pixels), args.warmup, args.iters)
                    torch_tower = float("nan")
                    if not args.no_torch_tower:
                        def _torch_tower():
                            with torch.autocast("cuda", dtype=torch.float16):
                                model.vision_model(pixels)
                        torch_tower = gpu_ms(_torch_tower, args.warmup, args.iters)

                    def _detect(px):
                        with torch.autocast("cuda", dtype=torch.float16):
                            return detect(model, px, token_ids, attention_mask)

                    det = gpu_ms(lambda: _detect(pixels), args.warmup, args.iters)

                    def _e2e():
                        px = model.preprocess_image(images, fast=fast).to(device)
                        logits, _, _ = _detect(px)
                        logits.float().cpu()

                    e2e = cpu_ms(_e2e, 1, prep_iters)

                torch_peak = torch.cuda.max_memory_allocated(device) / 2**30
                used = device_used_gib(device) - baseline_gib
                print(
                    f"{batch_size:>2} {prep_name:>5} | {preproc:8.1f} {h2d:6.1f} {trt_tower:8.1f} "
                    f"{torch_tower:9.1f} {det:7.1f} {e2e:7.1f} | "
                    f"{batch_size / e2e * 1e3:6.2f} {batch_size / trt_tower * 1e3:9.2f} | "
                    f"{torch_peak:8.2f} {used:8.2f}"
                )
            except torch.cuda.OutOfMemoryError as exc:
                print(f"{batch_size:>2} {prep_name:>5} | OOM: {str(exc).splitlines()[0]}")
                torch.cuda.empty_cache()
            except RuntimeError as exc:
                # TRT reports allocation failure through execute_async_v3 -> False.
                print(f"{batch_size:>2} {prep_name:>5} | FAILED: {str(exc).splitlines()[0]}")
                torch.cuda.empty_cache()

    print("\ntimes are ms per batch; img/s = B / e2e; memory in GiB above the idle device")


if __name__ == "__main__":
    main()
