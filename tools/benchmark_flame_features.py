"""Compare FLAME SVM backends, APE selection and visual feature sources.

Default support is sampled from val2017 outside the existing mini500 eval set,
since the local COCO installation has no train2017 annotations. Evaluation
images are excluded from fitting; C and alpha stay fixed. Timings exclude OWLv2
proposal extraction, but include transfers and feature construction during fit.

    .venv/bin/python tools/benchmark_flame_features.py --device cuda:1 \
        --components 256 1024 --seeds 0 --output data/flame_features/results.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from OWLv2torch import Flame, OwlV2
from OWLv2torch.torch_version.flame import select_detections


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def metrics(gt, detections):
    from faster_coco_eval import COCO, COCOeval_faster

    truth = COCO(gt)
    predicted = truth.loadRes(detections) if detections else COCO({**gt, "annotations": []})
    evaluator = COCOeval_faster(truth, predicted, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]

    def mean(array):
        valid = array[array > -1]
        return float(valid.mean()) if valid.size else None

    result = {"map": mean(precision[:, :, :, 0, -1]),
              "map_50": mean(precision[0, :, :, 0, -1]),
              "map_75": mean(precision[5, :, :, 0, -1])}
    result.update({f"map_{name}": mean(precision[:, :, :, i, -1])
                   for i, name in enumerate(("small", "medium", "large"), 1)})
    result["per_class"] = {category["name"]: mean(precision[:, :, i, 0, -1])
                           for i, category in enumerate(gt["categories"])}
    return result


def add_detections(records, image_id, boxes, scores):
    for box, score, label in zip(*select_detections(boxes, scores, threshold=0.0)):
        x1, y1, x2, y2 = box.tolist()
        records.append({"image_id": image_id, "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco-root", type=Path, default=Path("/mnt/datasets/coco"))
    parser.add_argument("--dataset-label", default="coco")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--eval-annotations", type=Path)
    parser.add_argument("--support-image-root", type=Path)
    parser.add_argument("--eval-image-root", type=Path)
    parser.add_argument("--query-template", default="a photo of a {name}")
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--support-images", type=int, default=5)
    parser.add_argument("--support-manifest", type=Path,
                        help="Optional JSON {seed: [image IDs]} to compare nested support sets.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--components", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--backends", nargs="+", choices=["sklearn", "torch", "nystrom", "rff"],
                        default=["nystrom", "rff"], help="sklearn is always included as the control.")
    parser.add_argument("--ape-channels", type=int, nargs="+", default=[],
                        help="Also fit sklearn FLAME after APE selects these channel budgets.")
    parser.add_argument("--ape-sources", nargs="+", choices=["text", "text_support"], default=["text_support"])
    parser.add_argument("--ape-similarity-weights", type=float, nargs="+", default=[0.9])
    parser.add_argument("--ape-normalize", action="store_true",
                        help="L2-normalize selected APE features before the SVM standardization.")
    parser.add_argument("--feature-sources", nargs="+", choices=["class", "visual", "combined"], default=["class"],
                        help="Also compare sklearn on pre-projection visual or combined features.")
    parser.add_argument("--visual-weights", type=float, nargs="+", default=[0.25, 0.5, 0.75],
                        help="Visual branch weights for combined standardized RBF distances.")
    parser.add_argument("--visual-no-normalize", action="store_true",
                        help="Use raw pre-projection features instead of L2-normalizing their rows.")
    parser.add_argument("--svm-max-iter", type=int, default=300)
    parser.add_argument("--svm-tol", type=float, default=1e-3)
    parser.add_argument("--svm-cache-size-mb", type=float, default=256)
    parser.add_argument("--svm-working-set-size", type=int, default=512)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/flame_features"))
    parser.add_argument("--output", type=Path, default=Path("data/flame_features/results.json"))
    parser.add_argument("--reference-results", type=Path,
                        help="Reuse a completed sklearn control from the identical support/eval protocol.")
    args = parser.parse_args()
    if any(not math.isfinite(w) or not 0 < w < 1 for w in args.visual_weights):
        parser.error("--visual-weights must be between 0 and 1, exclusive")
    include_visual = any(s != "class" for s in args.feature_sources)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    annotations = args.annotations or args.coco_root / "annotations/instances_val2017.json"
    eval_annotations = args.eval_annotations or args.coco_root / "annotations/instances_val2017_mini500.json"
    full = json.loads(annotations.read_text())
    evaluation = json.loads(eval_annotations.read_text())
    support_root = args.support_image_root or args.coco_root / "val2017"
    eval_root = args.eval_image_root or args.coco_root / "val2017"
    support_images = {image["id"]: image for image in full["images"]}
    evaluation_images = {image["id"]: image for image in evaluation["images"]}
    if {c["id"]: c["name"] for c in full["categories"]} != {
        c["id"]: c["name"] for c in evaluation["categories"]
    }:
        parser.error("Support and evaluation category IDs/names differ")
    categories = sorted(full["categories"], key=lambda category: category["id"])
    if args.classes:
        unknown = set(args.classes) - {category["name"] for category in categories}
        if unknown:
            parser.error(f"Unknown classes: {sorted(unknown)}")
        categories = [category for category in categories if category["name"] in args.classes]
    to_query = {category["id"]: i for i, category in enumerate(categories)}
    names = [category["name"] for category in categories]
    # IDs are scoped to a split: DIOR and DOTA train/test reuse many IDs for
    # different files. Exclude actual evaluation paths, not bare image IDs.
    # Exclude the entire supplied evaluation split even when --limit is set.
    excluded_paths = {(eval_root / image["file_name"]).resolve() for image in evaluation["images"]}
    excluded = {image["id"] for image in full["images"]
                if (support_root / image["file_name"]).resolve() in excluded_paths}
    annotations_by_image = defaultdict(list)
    for ann in full["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)
    by_class = defaultdict(set)
    for image_id, anns in annotations_by_image.items():
        if image_id in excluded or any(ann.get("iscrowd", 0) for ann in anns):
            continue
        for ann in anns:
            if ann["category_id"] in to_query and ann["bbox"][2] > 0 and ann["bbox"][3] > 0:
                by_class[ann["category_id"]].add(image_id)
    support_sets = {}
    for seed in args.seeds:
        rng = random.Random(seed)
        selected = set()
        for category in categories:
            pool = sorted(by_class[category["id"]])
            selected.update(rng.sample(pool, min(args.support_images, len(pool))))
        support_sets[seed] = sorted(selected)
    if args.support_manifest:
        manifest = json.loads(args.support_manifest.read_text())
        eligible = set().union(*by_class.values())
        for seed in args.seeds:
            ids = manifest.get(str(seed))
            if not isinstance(ids, list) or not ids or any(type(image_id) is not int for image_id in ids):
                parser.error(f"Support manifest needs a nonempty list of integer IDs for seed {seed}")
            if len(ids) != len(set(ids)) or set(ids) - eligible:
                parser.error(f"Support manifest has duplicate or ineligible image IDs for seed {seed}")
            support_sets[seed] = sorted(ids)
    eval_images = sorted(evaluation["images"], key=lambda image: image["id"])
    if args.limit:
        eval_images = eval_images[:args.limit]
    eval_ids = {image["id"] for image in eval_images}
    gt = {**evaluation, "images": eval_images,
          "categories": [{**category, "id": to_query[category["id"]]} for category in categories],
          "annotations": [{**ann, "category_id": to_query[ann["category_id"]]}
                          for ann in evaluation["annotations"]
                          if ann["image_id"] in eval_ids and ann["category_id"] in to_query]}
    model = OwlV2("base").eval().to(device)

    def query(name):
        return args.query_template.format(name=name.strip().lower().replace("-", " ").replace("_", " "))

    source = Flame(model, [query(name) for name in names], alpha=args.alpha, box_refinement=False,
                   svm_c=args.svm_c, batch_size=args.batch_size, svm_tol=args.svm_tol,
                   svm_cache_size_mb=args.svm_cache_size_mb,
                   svm_working_set_size=args.svm_working_set_size)
    # Image features, boxes and objectness are query-independent. Cache with the
    # complete vocabulary so subsets and later seeds can reuse extraction.
    all_names = [category["name"] for category in sorted(full["categories"], key=lambda c: c["id"])]
    extractor = Flame(model, [query(name) for name in all_names], batch_size=args.batch_size)
    query_columns = [all_names.index(name) for name in names]
    selectors = {mode: copy.copy(source) for mode in set(args.feature_sources) | {"class"}}
    for mode, selector in selectors.items():
        selector.feature_source = mode
        selector.visual_normalize = not args.visual_no_normalize

    def cache_directory(image_root):
        cache = args.cache_dir / ("base_visual_float32_v1" if include_visual else "base_default_float32_v1")
        # Keep the original COCO cache usable. Custom datasets/prompts get an
        # explicit namespace containing the physical image root and vocabulary.
        legacy = (args.dataset_label == "coco" and args.annotations is None and args.eval_annotations is None
                  and image_root.resolve() == (args.coco_root / "val2017").resolve()
                  and args.query_template == "a photo of a {name}")
        if not legacy:
            identity = json.dumps({"image_root": str(image_root.resolve()), "queries": [query(n) for n in all_names]})
            cache /= hashlib.sha256(identity.encode()).hexdigest()[:16]
        cache.mkdir(parents=True, exist_ok=True)
        return cache

    caches = {"support": cache_directory(support_root), "eval": cache_directory(eval_root)}
    groups = [("support", support_images, support_root,
               sorted({image_id for support in support_sets.values() for image_id in support})),
              ("eval", evaluation_images, eval_root, sorted(eval_ids))]
    missing_count = sum(sum(not (caches[split] / f"{image_id}.pt").exists() for image_id in needed)
                        for split, _, _, needed in groups)
    print(f"{len(names)} classes; {len(eval_images)} eval images; support images "
          f"{ {seed: len(ids) for seed, ids in support_sets.items()} }; {missing_count} uncached", flush=True)
    extraction_start = time.perf_counter()
    for split, split_images, image_root, needed in groups:
        missing = [image_id for image_id in needed if not (caches[split] / f"{image_id}.pt").exists()]
        for start in tqdm(range(0, len(missing), args.batch_size), desc=f"cache {split}"):
            batch_ids = missing[start:start + args.batch_size]
            batch = []
            for image_id in batch_ids:
                with Image.open(image_root / split_images[image_id]["file_name"]) as image:
                    batch.append(image.convert("RGB"))
            extraction = extractor.proposal_features(batch) if include_visual else extractor.proposals(batch)
            for image_id, proposals in zip(batch_ids, extraction):
                torch.save(tuple(tensor.cpu().clone() for tensor in proposals), caches[split] / f"{image_id}.pt")
    extraction_seconds = time.perf_counter() - extraction_start

    def load(image_id, split="eval", feature_source="class"):
        cached = torch.load(caches[split] / f"{image_id}.pt", weights_only=True)
        if include_visual:
            embeds, visual, logits, objectness, boxes = cached
        else:
            embeds, logits, objectness, boxes = cached
            visual = None
        selected = selectors[feature_source].select_features(
            embeds.to(device), visual.to(device) if feature_source != "class" else None,
        )
        return selected, logits[:, query_columns].to(device), objectness.to(device), boxes.to(device)

    payload = {"protocol": {"model": "owlv2-base", "device": str(device),
                            "dataset": args.dataset_label,
                            "support_image_root": str(support_root), "eval_image_root": str(eval_root),
                            "query_template": args.query_template,
                            "support_eval_image_paths_disjoint": True,
                            "cache_directories": {key: str(value) for key, value in caches.items()},
                            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                            "annotations": str(annotations), "eval_annotations": str(eval_annotations),
                            "eval_image_ids": sorted(eval_ids), "support_image_ids": support_sets,
                            "support_images_per_class": args.support_images, "classes": names,
                            "support_manifest": str(args.support_manifest) if args.support_manifest else None,
                            "alpha": args.alpha, "svm_c": args.svm_c,
                            "svm_max_iter": args.svm_max_iter,
                            "svm_tol": args.svm_tol, "svm_cache_size_mb": args.svm_cache_size_mb,
                            "svm_working_set_size": args.svm_working_set_size,
                            "ape_channels": args.ape_channels, "ape_sources": args.ape_sources,
                            "ape_similarity_weights": args.ape_similarity_weights,
                            "ape_normalize": args.ape_normalize,
                            "feature_sources": args.feature_sources, "visual_weights": args.visual_weights,
                            "visual_normalize": not args.visual_no_normalize,
                            "feature_cache_kind": "class_and_visual" if include_visual else "class",
                            "smoothing": 0.01 if any(b in {"nystrom", "rff"} for b in args.backends) else 0.0,
                            "extraction_seconds_this_run": extraction_seconds,
                            "approximate_solver": "newton", "exact_solver": "hierarchical_smo"}, "runs": []}
    previous = None
    if args.reference_results:
        previous = json.loads(args.reference_results.read_text())
        if include_visual and previous["protocol"].get("cache_directories") != payload["protocol"]["cache_directories"]:
            parser.error("A visual-feature comparison must reuse a control from the same representation cache")
        for key in ("model", "annotations", "eval_annotations", "eval_image_ids", "classes", "alpha", "svm_c"):
            if previous["protocol"][key] != payload["protocol"][key]:
                parser.error(f"Reference protocol differs for {key}")
        legacy_defaults = {"query_template": "a photo of a {name}",
                           "support_image_root": str(args.coco_root / "val2017"),
                           "eval_image_root": str(args.coco_root / "val2017")}
        for key, default in legacy_defaults.items():
            if previous["protocol"].get(key, default) != payload["protocol"][key]:
                parser.error(f"Reference protocol differs for {key}")
        for seed, ids in support_sets.items():
            if previous["protocol"]["support_image_ids"].get(str(seed)) != ids:
                parser.error(f"Reference support images differ for seed {seed}")
            if not any(run["backend"] == "sklearn" and run["seed"] == seed
                       and run.get("ape_channels") is None and run.get("feature_source", "class") == "class"
                       for run in previous["runs"]):
                parser.error(f"Reference has no completed sklearn control for seed {seed}")
        payload["reference_results"] = str(args.reference_results)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    if previous:
        payload["zero_shot"] = previous["zero_shot"]
    else:
        zero_shot = []
        for image in tqdm(eval_images, desc="zero-shot"):
            with torch.inference_mode():
                _, logits, objectness, boxes = load(image["id"])
                add_detections(zero_shot, image["id"], boxes,
                               torch.sigmoid(logits) * torch.sigmoid(objectness)[:, None])
        payload["zero_shot"] = metrics(gt, zero_shot)
    save()

    for seed, support in support_sets.items():
        example_banks = {}
        for mode, selector in selectors.items():
            selector.examples = [[] for _ in names]
            selector._rng = torch.Generator().manual_seed(0)
            for image_id in tqdm(support, desc=f"support {mode} seed {seed}"):
                anns = [ann for ann in annotations_by_image[image_id] if ann["category_id"] in to_query]
                boxes = [[x, y, x + w, y + h] for x, y, w, h in (ann["bbox"] for ann in anns)]
                labels = [to_query[ann["category_id"]] for ann in anns]
                with torch.inference_mode():
                    selector._add_proposals(load(image_id, "support", mode), boxes, labels)
            example_banks[mode] = selector.examples
        source.examples = example_banks["class"]
        for mode, bank in example_banks.items():
            for original, selected in zip(source.examples, bank):
                original_targets = torch.cat([y for _, y in original]) if original else torch.empty(0)
                selected_targets = torch.cat([y for _, y in selected]) if selected else torch.empty(0)
                if not torch.equal(original_targets, selected_targets):
                    raise RuntimeError(f"Support labels changed with feature source {mode}")
        counts = []
        for examples in source.examples:
            y = torch.cat([targets for _, targets in examples]) if examples else torch.empty(0)
            counts.append({"positive": int((y == 1).sum()), "negative": int((y == 0).sum())})
        variants = [(backend, rank, None, None, None, "class", None) for backend in args.backends if backend != "sklearn"
                    for rank in ([0] if backend == "torch" else args.components)]
        variants += [("sklearn", 0, channels, weight, ape_source, "class", None)
                     for ape_source in args.ape_sources for weight in args.ape_similarity_weights
                     for channels in args.ape_channels]
        if "visual" in args.feature_sources:
            variants.append(("sklearn", 0, None, None, None, "visual", None))
        if "combined" in args.feature_sources:
            variants += [("sklearn", 0, None, None, None, "combined", w) for w in args.visual_weights]
        if previous:
            reused = copy.deepcopy(next(run for run in previous["runs"]
                                        if run["backend"] == "sklearn" and run["seed"] == seed
                                        and run.get("ape_channels") is None
                                        and run.get("feature_source", "class") == "class"))
            reused["reused_from"] = str(args.reference_results)
            payload["runs"].append(reused)
            save()
        else:
            variants.insert(0, ("sklearn", 0, None, None, None, "class", None))
        reference = None
        for backend, rank, channels, weight, ape_source, feature_source, visual_weight in variants:
            flame = copy.copy(source)
            flame.svms = [None] * len(names)
            flame.svm_backend, flame.n_components = backend, rank
            flame.svm_seed, flame.svm_max_iter = seed, args.svm_max_iter
            flame.ape_channels = channels
            flame.feature_source = feature_source
            flame.visual_weight = 0.5 if visual_weight is None else visual_weight
            flame.visual_normalize = not args.visual_no_normalize
            flame.examples = example_banks[feature_source]
            if channels is not None:
                flame.ape_similarity_weight, flame.ape_source = weight, ape_source
                flame.ape_normalize = args.ape_normalize
            variant = f"{backend}/{rank}" if channels is None else f"ape/{ape_source}/{channels}/w{weight:g}"
            if feature_source != "class":
                variant = f"{feature_source}" + (f"/w{visual_weight:g}" if visual_weight is not None else "")
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                allocated_before = torch.cuda.memory_allocated(device)
            start = time.perf_counter()
            print(f"Fitting seed={seed} {variant} ...", flush=True)
            flame.fit()
            synchronize(device)
            fit_seconds = time.perf_counter() - start
            peak_mb = ((torch.cuda.max_memory_allocated(device) - allocated_before) / 2**20
                       if device.type == "cuda" else None)
            classifier_sizes = [sum(v.numel() * v.element_size() for v in svm.values() if torch.is_tensor(v))
                                for svm in flame.svms if svm is not None]
            if flame.ape_indices is not None:
                classifier_sizes.append(flame.ape_indices.numel() * flame.ape_indices.element_size())
            print(f"seed={seed} {variant}: fitted in {fit_seconds:.2f}s; "
                  f"classifier {sum(classifier_sizes) / 2**20:.2f} MiB", flush=True)
            detections, sampled_margins = [], []
            synchronize(device)
            start = time.perf_counter()
            for image in tqdm(eval_images, desc=f"{variant} seed {seed}"):
                with torch.inference_mode():
                    embeds, logits, objectness, boxes = load(image["id"], feature_source=feature_source)
                    margins = flame.margins(embeds)
                    # Sample high-objectness proposals to avoid measuring only background.
                    sample = objectness.topk(min(64, len(objectness))).indices
                    sampled_margins.append(margins[sample].cpu())
                    scores = torch.sigmoid(logits + flame.alpha * margins) * torch.sigmoid(objectness)[:, None]
                    add_detections(detections, image["id"], boxes, scores)
            synchronize(device)
            scoring_seconds = time.perf_counter() - start
            current = torch.cat(sampled_margins)
            if backend == "sklearn" and channels is None and feature_source == "class":
                reference = current
            difference = current - reference if reference is not None else None
            result = {"seed": seed, "backend": backend, "components": rank, "fit_seconds": fit_seconds,
                      "ape_channels": channels, "ape_similarity_weight": weight, "ape_source": ape_source,
                      "ape_normalize": args.ape_normalize if channels is not None else None,
                      "ape_indices": flame.ape_indices.tolist() if flame.ape_indices is not None else None,
                      "feature_source": feature_source, "feature_dim": flame.feature_dim,
                      "visual_weight": visual_weight, "visual_normalize": not args.visual_no_normalize,
                      "score_seconds_including_cache_io": scoring_seconds, "peak_fit_extra_mb": peak_mb,
                      "classifier_mb": sum(classifier_sizes) / 2**20, "support_counts": counts,
                      "missing_classes": [name for name, svm in zip(names, flame.svms) if svm is None],
                      "margin_mae": float(difference.abs().mean()) if difference is not None else None,
                      "margin_max_abs": float(difference.abs().max()) if difference is not None else None,
                      "margin_sign_disagreement": (float(((current > 0) != (reference > 0)).float().mean())
                                                   if reference is not None else None),
                      "solver": [svm.get("fit") if svm else None for svm in flame.svms],
                      "metrics": metrics(gt, detections)}
            payload["runs"].append(result)
            save()
            print(f"RESULT seed={seed} {variant}: mAP={result['metrics']['map']:.6f}, "
                  f"AP50={result['metrics']['map_50']:.6f}, fit={fit_seconds:.2f}s, "
                  f"margin MAE={result['margin_mae']}", flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
