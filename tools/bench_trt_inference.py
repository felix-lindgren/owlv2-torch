"""Benchmark OWLv2 TensorRT inference stage by stage.

Splits a single detection call into decode -> preprocess -> vision tower (TRT)
-> heads -> postprocess -> host copy, and profiles the kernels inside the TRT
engine to show where the vision tower spends its time.

    uv run python tools/bench_trt_inference.py --model-type large \
        --output-dir artifacts
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OWLv2torch import tokenize
from OWLv2torch.torch_version.owlv2_tensorrt import OwlV2TRT

QUERIES = [
    "a cat", "a dog", "a person", "a car", "a bicycle", "a plastic bag",
    "a scale", "a bottle", "a chair", "a table", "a ship", "an airplane",
    "a truck", "a bird", "a laptop", "a phone",
]


def cpu_timed(fn, iters, warmup=2):
    """Median wall-clock ms, synchronizing the GPU around each call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


def gpu_timed(fn, iters, warmup=5):
    """Median device ms per call from CUDA events (excludes host overhead gaps)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def kernel_category(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("fmha", "flash", "mha", "attention", "softmax")):
        return "attention (fused MHA)"
    if any(k in n for k in ("gemm", "cutlass", "xmma", "sm80_", "sm89_", "sm90_", "ampere", "hopper", "matmul")):
        return "GEMM (qkv/out/mlp)"
    if "conv" in n:
        return "conv (patch embed)"
    if any(k in n for k in ("layernorm", "layer_norm", "norm")):
        return "layernorm"
    if any(k in n for k in ("copy", "memcpy", "reformat", "transpose", "shuffle", "concat", "slice")):
        return "copies/reformat"
    return "other (elementwise/fused)"


def profile_kernels(fn, iters=5):
    """Sum CUDA kernel time per category over ``iters`` calls using torch.profiler."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()

    by_cat = defaultdict(float)
    by_kernel = defaultdict(lambda: [0.0, 0])
    for evt in prof.events():
        if evt.device_type != torch.autograd.DeviceType.CUDA:
            continue
        us = evt.device_time
        by_cat[kernel_category(evt.name)] += us / iters
        by_kernel[evt.name][0] += us / iters
        by_kernel[evt.name][1] += 1
    return by_cat, by_kernel


def fmt_row(name, ms, total=None):
    share = f"{ms / total * 100:6.1f}%" if total else ""
    return f"  {name:44s} {ms:9.2f} ms {share}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", default="base", choices=["base", "large"])
    p.add_argument("--output-dir", default=".", help="where OwlV2TRT looks for default-named engines")
    p.add_argument("--engine", default=None)
    p.add_argument("--heads-engine", default=None)
    p.add_argument("--no-trt-heads", action="store_true")
    p.add_argument("--image", default="img.jpg")
    p.add_argument("--batch-sizes", default="1,2,4,8")
    p.add_argument("--num-queries", type=int, default=16)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--score-threshold", type=float, default=0.1)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--skip-torch", action="store_true", help="skip eager torch vision comparison")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    model = OwlV2TRT(engine_path=args.engine, heads_engine_path=args.heads_engine,
                     trt_heads=not args.no_trt_heads, output_dir=args.output_dir,
                     model_type=args.model_type, build_missing=False)
    if model.trt is None:
        raise SystemExit(f"TRT engine not loaded from {model.engine_path}")
    model.eval().to(device)
    S = model.image_size
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    print(f"GPU: {torch.cuda.get_device_name()} | model={args.model_type} S={S} "
          f"tokens={model.sqrt_num_patches**2 + 1} engine={model.engine_path} "
          f"heads={model.heads_engine_path if model.trt_heads is not None else 'torch'}")

    # ---------------------------------------------------------------- preprocess
    img_path = Path(args.image)
    pil = Image.open(img_path).convert("RGB")
    print(f"\n[preprocess] per image, source {pil.size[0]}x{pil.size[1]} ({img_path.name})")

    def decode():
        im = Image.open(img_path)
        im.load()
        return im.convert("RGB")

    pre = {}
    pre["decode jpeg (PIL)"] = cpu_timed(decode, 10)
    pre["fast CPU resize + H2D (default)"] = cpu_timed(
        lambda: model.preprocess_image(pil).to(device), 10)
    pre["fast GPU resize (device='cuda')"] = cpu_timed(
        lambda: model.preprocess_image(pil, device=device), 20)
    pre["scipy reference (fast=False) + H2D"] = cpu_timed(
        lambda: model.preprocess_image(pil, fast=False).to(device), 3, warmup=1)
    for k, v in pre.items():
        print(fmt_row(k, v))

    # ---------------------------------------------------------------- text
    tokens = tokenize(QUERIES[: args.num_queries] * (args.num_queries // len(QUERIES) + 1),
                      context_length=16, truncate=True)[: args.num_queries].to(device)
    with torch.inference_mode():
        text_ms = gpu_timed(lambda: model.encode_detection_queries(tokens), args.iters)
        query_embeds, query_mask = model.encode_detection_queries(tokens)
    print(f"\n[text] encode {args.num_queries} queries (cacheable, once per query set): {text_ms:.2f} ms")

    # ---------------------------------------------------------------- per-stage by batch
    def heads(vision_full, autocast=False):
        with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
            image_feats, _ = model._detection_image_features_from_vision(vision_full)
            text = query_embeds.unsqueeze(0).expand(vision_full.shape[0], -1, -1)
            mask = query_mask.unsqueeze(0).expand(vision_full.shape[0], -1)
            logits, _ = model.class_head(image_feats, text.to(image_feats.dtype), mask)
            obj = model.objectness_head(image_feats)[..., 0]
            boxes = torch.sigmoid(model.box_head(image_feats) + model.box_bias)
        return logits, obj, boxes

    def postprocess(logits, obj, boxes, target_sizes):
        boxes_xyxy = model.postprocess_boxes(boxes.float(), target_sizes)
        scores = torch.sigmoid(logits.float()) * torch.sigmoid(obj.float()).unsqueeze(-1)
        max_scores, labels = scores.max(dim=-1)
        out = []
        for b in range(max_scores.shape[0]):
            s, l, bx = max_scores[b], labels[b], boxes_xyxy[b]
            keep = s >= args.score_threshold  # data-dependent -> host sync
            s, l, bx = s[keep], l[keep], bx[keep]
            if s.numel() > args.top_k:
                idx = torch.topk(s, args.top_k).indices
                s, l, bx = s[idx], l[idx], bx[idx]
            out.append((bx.cpu().numpy(), s.cpu().numpy(), l.cpu().numpy()))
        return out

    rows = []
    with torch.inference_mode():
        for B in batch_sizes:
            pixels = model.preprocess_image(pil, device=device).expand(B, -1, -1, -1).contiguous()
            target_sizes = torch.tensor([[pil.size[1], pil.size[0]]] * B, dtype=torch.float32, device=device)
            _, vision_full = model._get_vision_outputs(pixels)
            vision_full = vision_full.clone()
            outs = heads(vision_full)

            r = {"B": B}
            r["vision TRT"] = gpu_timed(lambda: model._get_vision_outputs(pixels), args.iters)
            r["heads fp32"] = gpu_timed(lambda: heads(vision_full), args.iters)
            r["heads fp16 autocast"] = gpu_timed(lambda: heads(vision_full, True), args.iters)
            if model.trt_heads is not None:
                r["heads TRT"] = gpu_timed(
                    lambda: model.trt_heads(vision_full=vision_full, query_embeds=query_embeds), args.iters)
            r["postprocess + D2H"] = cpu_timed(lambda: postprocess(*outs, target_sizes), args.iters)

            def e2e(gpu_pre):
                if gpu_pre:
                    px = model.preprocess_image([pil] * B, device=device)
                else:
                    px = model.preprocess_image([pil] * B).to(device)
                # The shipped path: TRT heads when the engine is loaded, torch heads otherwise.
                o = model.forward_object_detection_from_embeddings(px, query_embeds, query_mask)
                return postprocess(*o[:3], target_sizes)

            r["e2e (GPU preprocess)"] = cpu_timed(lambda: e2e(True), max(5, args.iters // 3))
            r["e2e (CPU preprocess)"] = cpu_timed(lambda: e2e(False), 5, warmup=1)
            r["model only"] = cpu_timed(
                lambda: model.forward_object_detection_from_embeddings(pixels, query_embeds, query_mask),
                args.iters)
            if not args.skip_torch:
                with torch.autocast("cuda", dtype=torch.float16):
                    r["vision torch eager fp16"] = gpu_timed(lambda: model.vision_model(pixels), 10)
            rows.append(r)

    for r in rows:
        B = r["B"]
        print(f"\n[pipeline] B={B}  (ms per batch; per-image = /{B})")
        e2e = r["e2e (GPU preprocess)"]
        print(fmt_row("vision TRT", r["vision TRT"], e2e))
        if "heads TRT" in r:
            print(fmt_row("heads TRT", r["heads TRT"], e2e))
        print(fmt_row("heads torch fp32", r["heads fp32"], e2e))
        print(fmt_row("heads torch fp16 autocast", r["heads fp16 autocast"], e2e))
        print(fmt_row("postprocess + D2H", r["postprocess + D2H"], e2e))
        print(fmt_row("model only (vision + heads, wall)", r["model only"], e2e))
        print(fmt_row("E2E PIL->host dets, GPU preprocess", e2e, e2e))
        print(fmt_row("E2E PIL->host dets, CPU preprocess", r["e2e (CPU preprocess)"]))
        if "vision torch eager fp16" in r:
            v = r["vision torch eager fp16"]
            print(fmt_row(f"[ref] vision torch eager fp16 ({v / r['vision TRT']:.2f}x TRT)", v))
        print(f"  throughput: {B / e2e * 1e3:.1f} img/s e2e, "
              f"{B / r['vision TRT'] * 1e3:.1f} img/s vision-only")

    # ---------------------------------------------------------------- inside the engine
    B = batch_sizes[0]
    pixels = model.preprocess_image(pil, device=device).expand(B, -1, -1, -1).contiguous()
    with torch.inference_mode():
        by_cat, by_kernel = profile_kernels(lambda: model._get_vision_outputs(pixels))
    total = sum(by_cat.values())
    print(f"\n[vision TRT kernels] B={B}, summed kernel time {total / 1e3:.2f} ms")
    for cat, us in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(fmt_row(cat, us / 1e3, total / 1e3))
    print("  top kernels:")
    for name, (us, n) in sorted(by_kernel.items(), key=lambda kv: -kv[1][0])[:12]:
        print(f"    {us / 1e3:7.2f} ms  {n // 5:4d}x/call  {name[:110]}")

    with torch.inference_mode():
        _, vision_full = model._get_vision_outputs(pixels)
        by_cat, by_kernel = profile_kernels(lambda: heads(vision_full))
    total = sum(by_cat.values())
    print(f"\n[heads torch fp32 kernels] B={B}, summed kernel time {total / 1e3:.2f} ms")
    for name, (us, n) in sorted(by_kernel.items(), key=lambda kv: -kv[1][0])[:10]:
        print(f"    {us / 1e3:7.2f} ms  {n // 5:4d}x/call  {name[:110]}")

    print(f"\npeak memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB (torch allocator only)")


if __name__ == "__main__":
    main()
