"""Measure what the per-step text path costs in ``prototype_train/train_text.py``.

Splits the cost into the CPU BPE tokenization, the host-to-device copy and the
text-tower forward, then compares it against a full training step at the same
config so the saving from precomputing embeddings can be read off directly.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OWLv2torch.torch_version.owlv2 import OwlV2
from OWLv2torch.torch_version.text_loss import compute_text_query_losses
from OWLv2torch.utils.tokenizer import tokenize
from prototype_train.train_text import (
    DEFAULT_PROMPT_TEMPLATES,
    build_prompt_pools,
    configure_trainable_parameter_groups,
    sample_prompt_set,
)

FASHIONPEDIA_CLASSES = [
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket",
    "vest", "pants", "shorts", "skirt", "coat", "dress", "jumpsuit", "cape",
    "glasses", "hat", "headband, head covering, hair accessory", "tie", "glove",
    "watch", "belt", "leg warmer", "tights, stockings", "sock", "shoe",
    "bag, wallet", "scarf", "umbrella", "hood", "collar", "lapel", "epaulette",
    "sleeve", "pocket", "neckline", "buckle", "zipper", "applique", "bead",
    "bow", "flower", "fringe", "ribbon", "rivet", "ruffle", "sequin", "tassel",
]


def timed(fn, *, repeats: int, warmup: int, sync: bool) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=("base", "large"), default="large")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vision-blocks", type=int, default=2)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--step-repeats", type=int, default=8)
    parser.add_argument("--skip-full-step", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    class_names = FASHIONPEDIA_CLASSES
    pools = build_prompt_pools(class_names, list(DEFAULT_PROMPT_TEMPLATES))
    generator = random.Random(0)
    unique_prompts = sorted({p for pool in pools for p in pool})
    print(f"classes={len(class_names)}  unique prompts in pool={len(unique_prompts)}")

    model = OwlV2(args.model_type).to(device).eval()

    # 1. CPU tokenization, exactly as TextQueryDetector.forward does it per step.
    def do_tokenize():
        prompts = sample_prompt_set(pools, generator)
        return tokenize(prompts, context_length=16, truncate=True)

    tok_ms, tok_sd = timed(do_tokenize, repeats=args.repeats, warmup=5, sync=False)
    print(f"tokenize({len(class_names)} prompts)            {tok_ms:8.3f} ms  (sd {tok_sd:.3f})")

    # Cost of tokenizing the whole pool once, for the precompute path.
    pool_ms, _ = timed(
        lambda: tokenize(unique_prompts, context_length=16, truncate=True),
        repeats=5, warmup=1, sync=False,
    )
    print(f"tokenize(full pool, one-off)          {pool_ms:8.3f} ms")

    token_ids_cpu = do_tokenize()

    def do_copy():
        return token_ids_cpu.to(device)

    copy_ms, _ = timed(do_copy, repeats=args.repeats, warmup=5, sync=True)
    print(f"token ids H2D copy                    {copy_ms:8.3f} ms")

    token_ids = token_ids_cpu.to(device)
    attention_mask = token_ids == 0

    def do_text_forward():
        with torch.autocast(device_type=device.type, enabled=True):
            return model.get_text_features(token_ids, attention_mask)

    text_ms, text_sd = timed(do_text_forward, repeats=args.repeats, warmup=10, sync=True)
    print(f"text tower forward (amp)              {text_ms:8.3f} ms  (sd {text_sd:.3f})")

    # The indexing that replaces all of the above in the precompute path.
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=True):
        pool_token_ids = tokenize(unique_prompts, context_length=16, truncate=True).to(device)
        pool_embeds = model.get_text_features(pool_token_ids, pool_token_ids == 0).float()
    index = torch.randint(0, len(unique_prompts), (len(class_names),), device=device)

    def do_gather():
        return pool_embeds.index_select(0, index)

    gather_ms, _ = timed(do_gather, repeats=args.repeats, warmup=10, sync=True)
    print(f"precomputed index_select (replacement){gather_ms:8.3f} ms")

    total = tok_ms + copy_ms + text_ms
    print(f"\nper-step text path total              {total:8.3f} ms")
    print(f"per-step after precompute             {gather_ms:8.3f} ms")
    print(f"saving per step                       {total - gather_ms:8.3f} ms")

    if args.skip_full_step:
        return

    # 2. A full training step at the same config, for the denominator.
    class _Args:
        head_learning_rate = 1e-5
        vision_learning_rate = 5e-6
        text_learning_rate = 1e-7
        train_box_head = True
        train_objectness_head = True
        vision_blocks = args.vision_blocks
        text_blocks = 0

    groups = configure_trainable_parameter_groups(model, _Args())
    if args.grad_checkpointing:
        model.set_gradient_checkpointing(True)
    model.train()
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=True)

    images = torch.randn(args.batch_size, 3, model.image_size, model.image_size, device=device)
    targets = [
        {
            "boxes": torch.tensor([[0.2, 0.2, 0.4, 0.4], [0.5, 0.5, 0.7, 0.8]], device=device),
            "labels": torch.tensor([3, 11], device=device),
        }
        for _ in range(args.batch_size)
    ]

    def do_step():
        prompts = sample_prompt_set(pools, generator)
        ids = tokenize(prompts, context_length=16, truncate=True).to(device)
        with torch.autocast(device_type=device.type, enabled=True):
            outputs = model.forward_object_detection(images, ids, ids == 0)
            losses = compute_text_query_losses(outputs, targets)
        scaler.scale(losses["loss"]).backward()
        for value in losses.values():
            float(value.detach())  # the per-step sync the trainer already does
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    step_ms, step_sd = timed(do_step, repeats=args.step_repeats, warmup=3, sync=True)
    print(
        f"\nfull train step (bs{args.batch_size}, vb{args.vision_blocks}, "
        f"ckpt={args.grad_checkpointing})  {step_ms:8.1f} ms  (sd {step_sd:.1f})"
    )
    print(f"text path share of step               {100 * total / step_ms:8.2f} %")
    saving_s_per_1000 = (total - gather_ms) * 1000 / 1000.0
    print(f"saving over 1000 steps                {saving_s_per_1000:8.2f} s")
    print(f"1000-step run: {step_ms:.0f} ms/step -> {step_ms * 1000 / 1000 / 60:.1f} min total")


if __name__ == "__main__":
    main()
