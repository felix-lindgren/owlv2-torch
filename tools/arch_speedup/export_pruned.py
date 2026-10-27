"""Export, build, check and time a padding-free (optionally pruned) OWLv2 TensorRT engine.

The engine takes images with their square padding cropped off (any multiple of
the patch size up to 960) plus shared query embeddings, and returns detections
for the tokens that reach the heads; see ``export.PaddingFreeDetector``. With
``--scorer`` (from ``distill_scorer.py``) the tower prunes after the scorer's
block to ``--keep`` of the full grid. The original weights are used throughout.

python tools/arch_speedup/export_pruned.py --scorer text_checkpoints/zeroshot_scorers/objectness_scorer_b4.pt \
    --keep 0.5 --out trt_engines/owlv2_base_padfree_b4_k50
python tools/arch_speedup/export_pruned.py --out trt_engines/owlv2_base_padfree   # padding drop only
python tools/arch_speedup/export_pruned.py --model-type large --out trt_engines/owlv2_large_padfree
"""
import argparse
import time
from pathlib import Path

import torch
from PIL import Image

from OWLv2torch.torch_version.build_engine import build_detector_engine
from OWLv2torch.torch_version.export import export_padding_free_detector
from OWLv2torch.torch_version.owlv2 import OwlV2
from OWLv2torch.torch_version.owlv2_tensorrt import PaddingFreeDetectorTRT
from OWLv2torch.utils.tokenizer import tokenize

CHECK_IMAGES = ["/mnt/datasets/coco/val2017/000000000139.jpg", "/mnt/datasets/coco/val2017/000000000785.jpg"]


def padding_free(model, image):
    pixels = model.preprocess_image(image, fast=True, device="cuda")
    rows, cols = model.padding_free_grid([(image.height, image.width)])
    return pixels[..., : rows * model.patch_size, : cols * model.patch_size]


def check(model, engine, query_embeds):
    """Torch (fp16 autocast) against the engine on real images, matched by token index."""
    for path in CHECK_IMAGES:
        with Image.open(path) as image:
            pixels = padding_free(model, image.convert("RGB"))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            logits, objectness, boxes, _, _ = model.forward_object_detection_from_embeddings(pixels, query_embeds)
        cols = pixels.shape[-1] // model.patch_size
        prune = getattr(model.vision_model.encoder, "prune", None)
        if prune is not None:
            local = model.vision_model.prune_aux[1]
            index = local // cols * model.sqrt_num_patches + local % cols
        else:
            local = torch.arange(objectness.shape[1], device=pixels.device)[None]
            index = local // cols * model.sqrt_num_patches + local % cols
        t_logits, t_objectness, t_boxes, t_index = engine(pixels, query_embeds)
        torch.cuda.synchronize()
        ref = dict(zip(index[0].tolist(), range(index.shape[1])))
        shared = [(ref[i], j) for j, i in enumerate(t_index[0].tolist()) if i in ref]
        a, b = torch.tensor([s[0] for s in shared]), torch.tensor([s[1] for s in shared])
        overlap = len(shared) / max(len(ref), t_index.shape[1])
        d_obj = (objectness[0, a].float() - t_objectness[0, b]).abs().max().item()
        d_box = (boxes[0, a].float() - t_boxes[0, b]).abs().max().item()
        d_logit = (logits[0, a].float() - t_logits[0, b]).abs().max().item()
        top_t = t_objectness[0].topk(20).indices
        top_ref = {int(index[0, i]) for i in objectness[0].float().topk(20).indices}
        top20 = len({int(t_index[0, i]) for i in top_t} & top_ref) / 20
        print(f"{Path(path).name} {tuple(pixels.shape[-2:])}: {t_index.shape[1]} tokens, kept-set overlap {overlap:.3f}, "
              f"max |d| objectness {d_obj:.3f} logits {d_logit:.3f} boxes {d_box:.4f}, top-20 agreement {top20:.2f}")


def bench(engine, query_embeds, shapes, iters=50):
    for shape in shapes:
        pixels = torch.rand(shape, device="cuda")
        for _ in range(10):
            engine(pixels, query_embeds)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            engine(pixels, query_embeds)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) / iters * 1000
        print(f"  {str(shape):22s} {ms:7.2f} ms  ({ms / shape[0]:.2f} ms/image)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scorer", type=Path, default=None, help="distilled scorer; omit for padding drop only")
    p.add_argument("--keep", type=float, default=None, help="fraction of the full grid kept (default: the scorer's)")
    p.add_argument("--out", type=Path, required=True, help="path prefix for .onnx and .engine")
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--max-queries", type=int, default=1024)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--model-type", default="base", choices=["base", "large"])
    args = p.parse_args()

    model = OwlV2(args.model_type).eval()
    S, patch = model.image_size, model.patch_size
    if args.scorer is not None:
        model.load_token_scorer(args.scorer, keep=args.keep)
    onnx_path, engine_path = args.out.with_suffix(".onnx"), args.out.with_suffix(".engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    if args.rebuild or not engine_path.exists():
        start = time.time()
        export_padding_free_detector(model, str(onnx_path), max_batch=args.max_batch)
        print(f"exported {onnx_path} ({time.time() - start:.0f}s)")
        start = time.time()
        if build_detector_engine(str(onnx_path), str(engine_path), max_batch=args.max_batch,
                                 max_queries=args.max_queries, image_size=S, patch_size=patch,
                                 opt_hw=(S, S)) is None:
            raise SystemExit("engine build failed")
        print(f"built {engine_path} ({time.time() - start:.0f}s)")

    model.cuda()
    engine = PaddingFreeDetectorTRT(engine_path)
    tokens = tokenize([f"a photo of a {n}" for n in ("person", "dog", "car", "cup", "chair")],
                      context_length=16, truncate=True).cuda()
    with torch.no_grad():
        query_embeds = model.get_text_features(tokens, tokens == 0)
    check(model, engine, query_embeds)
    print("TensorRT latency, 80 queries:")
    bench(engine, query_embeds.repeat(16, 1), [(b, 3, h, S) for b, h in ((1, S), (1, S * 3 // 4 // patch * patch),
                                                                (1, S * 9 // 16 // patch * patch), (8, S), (8, S * 3 // 4 // patch * patch))])


if __name__ == "__main__":
    main()
