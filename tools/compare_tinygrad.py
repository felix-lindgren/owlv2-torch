"""Compare the tinygrad OWLv2 port against the torch reference.

Preprocesses one image with the torch model, runs the same pixels and queries
through both implementations, and reports per-output error plus whether the
top detections agree.

    uv run python tools/compare_tinygrad.py --model-type base --image img.jpg
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OWLv2torch import OwlV2, tokenize
from OWLv2torch.tinygrad_version import OwlV2 as OwlV2Tiny

QUERIES = ["a cat", "a dog", "a person", "a car", "a bottle", "a chair", "<padding>"]


def report(name, ref, out, mask=None):
    ref = np.asarray(ref, dtype=np.float64)
    out = np.asarray(out, dtype=np.float64)
    assert ref.shape == out.shape, f"{name}: shape {ref.shape} vs {out.shape}"
    if mask is not None:
        ref, out = ref[mask], out[mask]
    diff = np.abs(ref - out)
    rel = diff.max() / max(np.abs(ref).max(), 1e-12)
    print(f"  {name:22s} shape={str(tuple(np.shape(ref))):18s} max_abs={diff.max():.3e} "
          f"mean_abs={diff.mean():.3e} max_rel={rel:.3e}")
    return diff.max()


def top_k(logits, objectness, k):
    """Top-k (patch, query) pairs by max class logit, as the README post-processing ranks them."""
    best = logits.max(-1)
    idx = np.argsort(-best)[:k]
    return idx, logits.argmax(-1)[idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-type", choices=["base", "large"], default="base")
    parser.add_argument("--image", default=str(REPO_ROOT / "img.jpg"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="torch device for the reference model")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    # Keep the reference in true fp32 so the comparison isn't dominated by TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    ref_model = OwlV2(args.model_type).eval().to(args.device)
    tiny_model = OwlV2Tiny(args.model_type)

    pixel_values = ref_model.preprocess_image(Image.open(args.image).convert("RGB"))
    token_ids = tokenize(QUERIES, context_length=16, truncate=True)
    # Make the last query all padding (id 0) to exercise the query mask.
    token_ids[-1] = 0
    query_mask = (token_ids[:, 0] > 0).numpy()

    with torch.inference_mode():
        pv, ids = pixel_values.to(args.device), token_ids.to(args.device)
        t0 = time.perf_counter()
        ref_det = [t.cpu().numpy() if torch.is_tensor(t) else t
                   for t in ref_model.forward_object_detection(pv, ids)[:4]]
        t_ref = time.perf_counter() - t0
        ref_clip = [t.cpu().numpy() for t in ref_model.forward(pv, ids, None)[:4]]

    pv_np, ids_np = pixel_values.numpy(), token_ids.numpy()
    t0 = time.perf_counter()
    tiny_det = [t.numpy() for t in tiny_model.forward_object_detection(pv_np, ids_np)[:4]]
    t_tiny = time.perf_counter() - t0
    tiny_clip = [t.numpy() for t in tiny_model.forward(pv_np, ids_np)[:4]]

    print(f"model={args.model_type} image={args.image} queries={QUERIES}")
    print(f"first-call time (includes tinygrad kernel compile): torch {t_ref:.2f}s, tinygrad {t_tiny:.2f}s")

    # An all-padding query masks every key, which torch's CUDA SDPA and its CPU
    # path resolve differently, so CLIP outputs are compared on valid queries.
    print("CLIP forward (valid queries):")
    masks = [(..., query_mask), (query_mask,), None, (query_mask,)]
    for name, r, t, m in zip(["logits_per_image", "logits_per_text", "vision_features", "text_features"],
                             ref_clip, tiny_clip, masks):
        report(name, r, t, mask=m)

    print("object detection:")
    ref_logits, ref_obj, ref_boxes, ref_cls = ref_det
    tiny_logits, tiny_obj, tiny_boxes, tiny_cls = tiny_det
    report("pred_logits (valid q)", ref_logits, tiny_logits, mask=(..., query_mask))
    masked_ok = np.array_equal(ref_logits[..., ~query_mask], tiny_logits[..., ~query_mask])
    print(f"  {'pred_logits (padded q)':22s} identical fill value: {masked_ok}")
    report("objectness_logits", ref_obj, tiny_obj)
    report("pred_boxes", ref_boxes, tiny_boxes)
    report("class_embeds", ref_cls, tiny_cls)

    valid_ref, valid_tiny = ref_logits[0][:, query_mask], tiny_logits[0][:, query_mask]
    ref_idx, ref_lab = top_k(valid_ref, ref_obj[0], args.top_k)
    tiny_idx, tiny_lab = top_k(valid_tiny, tiny_obj[0], args.top_k)
    same = np.array_equal(ref_idx, tiny_idx) and np.array_equal(ref_lab, tiny_lab)
    overlap = len(set(ref_idx.tolist()) & set(tiny_idx.tolist()))
    print(f"top-{args.top_k} detections identical (order + label): {same}; patch overlap {overlap}/{args.top_k}")
    print(f"  best score torch {1 / (1 + np.exp(-valid_ref.max())):.6f} "
          f"tinygrad {1 / (1 + np.exp(-valid_tiny.max())):.6f}")


if __name__ == "__main__":
    main()
