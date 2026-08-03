"""Micro-benchmark the OWLv2 training step to find where the time actually goes.

Run on a free device, e.g. ``--device cuda:1``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OWLv2torch.torch_version.owlv2 import OwlV2
from OWLv2torch.torch_version.text_loss import compute_text_query_losses


def timed(fn, device, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) / iters


def make_targets(batch_size, num_classes, device, num_boxes=8):
    targets = []
    for _ in range(batch_size):
        cxcy = torch.rand(num_boxes, 2, device=device) * 0.6 + 0.2
        wh = torch.rand(num_boxes, 2, device=device) * 0.2 + 0.05
        targets.append(
            {
                "boxes": torch.cat([cxcy, wh], dim=1),
                "labels": torch.randint(0, num_classes, (num_boxes,), device=device),
            }
        )
    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--model-type", default="large")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=46)
    parser.add_argument("--vision-blocks", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    model = OwlV2(args.model_type).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for module in (model.class_head, model.box_head, model.objectness_head):
        for p in module.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    if args.vision_blocks:
        mods = [
            *model.vision_model.encoder.layers[-args.vision_blocks :],
            model.vision_model.post_layernorm,
            model.layer_norm,
        ]
        for module in mods:
            for p in module.parameters():
                p.requires_grad_(True)
                trainable.append(p)

    B = args.batch_size
    S = model.image_size
    P = model.sqrt_num_patches
    N = P * P + 1
    pixel_values = torch.randn(B, 3, S, S, device=device)
    token_ids = torch.randint(1, 49407, (args.num_classes, 16), device=device)
    token_ids[:, 0] = 49406
    targets = make_targets(B, args.num_classes, device)

    optimizer = torch.optim.AdamW(trainable, lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    print(f"device={device} model={args.model_type} B={B} tokens={N} classes={args.num_classes}")
    print(f"trainable params: {sum(p.numel() for p in trainable):,}")

    # ---- roofline reference: a big dense GEMM ----
    a = torch.randn(8192, 8192, device=device, dtype=torch.float16)
    b = torch.randn(8192, 8192, device=device, dtype=torch.float16)
    t = timed(lambda: torch.mm(a, b), device, iters=20)
    print(f"\n[roofline] 8192^3 fp16 mm: {t*1e3:.2f} ms -> {2*8192**3/t/1e12:.1f} TFLOPS")

    # ---- component timings ----
    autocast = lambda: torch.autocast("cuda")

    def vision_fwd_nograd():
        with torch.no_grad(), autocast():
            model.vision_model(pixel_values)

    def text_fwd():
        with torch.no_grad(), autocast():
            model.get_text_features(token_ids, None)

    def full_fwd():
        with torch.no_grad(), autocast():
            model.forward_object_detection(pixel_values, token_ids)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = model.forward_object_detection(pixel_values, token_ids)
            losses = compute_text_query_losses(
                outputs, targets, negative_ratio=20, max_negatives_per_image=512
            )
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()

    def fwd_and_loss_only():
        with autocast():
            outputs = model.forward_object_detection(pixel_values, token_ids)
            compute_text_query_losses(
                outputs, targets, negative_ratio=20, max_negatives_per_image=512
            )

    results = {}
    for name, fn in [
        ("vision tower fwd (no grad)", vision_fwd_nograd),
        ("text tower fwd", text_fwd),
        ("full detection fwd (no grad)", full_fwd),
        ("fwd + loss (grad on)", fwd_and_loss_only),
        ("FULL TRAIN STEP", train_step),
    ]:
        results[name] = timed(fn, device, iters=args.iters)
        print(f"  {name:32s} {results[name]*1e3:8.1f} ms")

    # ---- FLOP accounting for the vision tower ----
    cfg = model.vision_model.encoder.layers[0]
    d = cfg.self_attn.embed_dim
    mlp = cfg.mlp[0].out_features
    L = len(model.vision_model.encoder.layers)
    T = B * N
    proj = 4 * T * d * d * 2
    mlp_f = 2 * T * d * mlp * 2
    attn = 2 * B * N * N * d * 2
    per_layer = proj + mlp_f + attn
    total = per_layer * L
    tv = results["vision tower fwd (no grad)"]
    print(f"\n[vision tower] {L} layers, d={d}, mlp={mlp}, tokens/img={N}")
    print(f"  proj {proj/1e9:.0f} GF | mlp {mlp_f/1e9:.0f} GF | attn {attn/1e9:.0f} GF per layer")
    print(f"  attn share of layer FLOPs: {attn/per_layer*100:.0f}%")
    print(f"  total fwd {total/1e12:.1f} TFLOP in {tv*1e3:.0f} ms -> {total/tv/1e12:.1f} TFLOPS")

    # ---- which SDPA backend runs? ----
    q = torch.randn(B, cfg.self_attn.num_heads, N, cfg.self_attn.head_dim, device=device, dtype=torch.float16)
    from torch.nn.attention import SDPBackend, sdpa_kernel

    print("\n[sdpa backends] (B,H,N,D) =", tuple(q.shape))
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
        try:
            with sdpa_kernel(backend):
                t = timed(
                    lambda: torch.nn.functional.scaled_dot_product_attention(q, q, q, scale=0.125),
                    device,
                    iters=5,
                )
            print(f"  {backend.name:20s} {t*1e3:7.2f} ms")
        except Exception as exc:  # noqa: BLE001
            print(f"  {backend.name:20s} unavailable: {type(exc).__name__}")
    t = timed(
        lambda: torch.nn.functional.scaled_dot_product_attention(q, q, q, scale=0.125),
        device,
        iters=5,
    )
    print(f"  {'DEFAULT':20s} {t*1e3:7.2f} ms")

    print(f"\npeak memory: {torch.cuda.max_memory_allocated(device)/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
