"""Compare a small regularized box head with unchanged-box, Torch-fit FLAME.

Select settings on development images with --selection-output, then evaluate
the fixed --configs on held-out images. Reuses full class/visual proposal caches;
no image encoder passes or backbone updates are needed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from OWLv2torch import Flame, OwlV2
from benchmark_flame_features import add_detections, metrics, synchronize


def grid():
    return [dict(name=f"iou{iou:g}_ridge{ridge:g}_strength{strength:g}",
                 config=dict(ridge=ridge, min_iou=iou, strength=strength,
                             max_proposals_per_object=4, max_center_shift=0.25, max_log_scale=0.4))
            for iou in [0.3, 0.5] for ridge in [0.001, 0.01, 0.1] for strength in [0.5, 1.0]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, help="Saved Torch sampling benchmark for baseline agreement")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path)
    parser.add_argument("--configs", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--svm-max-iter", type=int, default=300)
    args = parser.parse_args()
    if args.configs and args.selection_output:
        parser.error("Select on development data, then evaluate fixed --configs on held-out data")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    reference = json.loads(args.reference_results.read_text())
    protocol = reference["protocol"]
    if protocol["model"] != "owlv2-base" or protocol.get("feature_cache_kind") != "class_and_visual":
        parser.error("Requires an owlv2-base reference with full class/visual caches")
    configurations = json.loads(args.configs.read_text())["configs"] if args.configs else grid()
    if not configurations or len({c["name"] for c in configurations}) != len(configurations):
        parser.error("Box configurations must be nonempty with unique names")
    if any(entry["name"] == "baseline" for entry in configurations):
        parser.error("baseline is reserved for the unchanged-box control")
    seeds = args.seeds or sorted(map(int, protocol["support_image_ids"]))
    full = json.loads(Path(protocol["annotations"]).read_text())
    evaluation = json.loads(Path(protocol["eval_annotations"]).read_text())
    names = protocol["classes"]
    categories = sorted(full["categories"], key=lambda c: c["id"])
    all_names = [c["name"] for c in categories]
    to_query = {c["id"]: names.index(c["name"]) for c in categories if c["name"] in names}
    query_columns = [all_names.index(name) for name in names]
    eval_ids = set(protocol["eval_image_ids"])
    eval_images = sorted((im for im in evaluation["images"] if im["id"] in eval_ids), key=lambda im: im["id"])
    if len(eval_images) != len(eval_ids) or not eval_images:
        parser.error("Reference evaluation IDs do not match annotations")
    selected_categories = sorted((c for c in categories if c["id"] in to_query), key=lambda c: to_query[c["id"]])
    gt = {**evaluation, "images": eval_images,
          "categories": [{**c, "id": to_query[c["id"]]} for c in selected_categories],
          "annotations": [{**a, "category_id": to_query[a["category_id"]]}
                          for a in evaluation["annotations"]
                          if a["image_id"] in eval_ids and a["category_id"] in to_query]}
    anns_by_image = defaultdict(list)
    for ann in full["annotations"]:
        if ann["category_id"] in to_query:
            anns_by_image[ann["image_id"]].append(ann)
    images = {im["id"]: im for im in full["images"]}
    support_ids = sorted({i for seed in seeds for i in protocol["support_image_ids"][str(seed)]})
    excluded_paths = {(Path(protocol["eval_image_root"]) / im["file_name"]).resolve()
                      for im in evaluation["images"]}
    for image_id in support_ids:
        if (Path(protocol["support_image_root"]) / images[image_id]["file_name"]).resolve() in excluded_paths:
            parser.error(f"Support/evaluation overlap: image {image_id}")
    if args.configs:
        selection = json.loads(args.configs.read_text())
        development_paths = set(selection.get("development_image_paths", []))
        current_paths = {str(path) for path in excluded_paths}
        if development_paths & current_paths:
            parser.error("Held-out images overlap the images used to select the box configuration")
    caches = {split: Path(path) for split, path in protocol["cache_directories"].items()}

    def load(image_id, split):
        tensors = torch.load(caches[split] / f"{image_id}.pt", weights_only=True, map_location="cpu")
        if len(tensors) != 5:
            raise ValueError("Box refinement requires the full class/visual proposal cache")
        embeds, visual, logits, objectness, boxes = tensors
        return embeds, visual, logits[:, query_columns], objectness, boxes

    support_records, box_records = {}, {}
    for image_id in tqdm(support_ids, desc="load support cache"):
        embeds, visual, logits, objectness, boxes = load(image_id, "support")
        anns = anns_by_image[image_id]
        truth = [[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in anns)]
        labels = [to_query[a["category_id"]] for a in anns]
        support_records[image_id] = ((embeds, logits, objectness, boxes), truth, labels)
        valid_truth = [b for a, b in zip(anns, truth) if not a.get("iscrowd", 0) and not a.get("ignore", 0)
                       and b[2] > b[0] and b[3] > b[1]]
        box_records[image_id] = (visual, boxes, valid_truth)

    model = OwlV2("base").eval().to(device)
    queries = [protocol["query_template"].format(name=n.strip().lower().replace("-", " ").replace("_", " "))
               for n in names]
    payload = dict(protocol={**protocol, "reference_results": str(args.reference_results),
                             "baseline_results": str(args.baseline_results) if args.baseline_results else None,
                             "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                             "threads": args.threads, "svm_backend": "torch", "evaluated_seeds": seeds,
                             "svm_max_iter": args.svm_max_iter, "feature_source": "class",
                             "feature_sources": ["class"], "visual_weights": [],
                             "extraction_seconds_this_run": 0.0,
                             "box_feature_source": "visual", "box_fit": "object-weighted ridge, regularized intercept",
                             "box_class_agnostic": True, "fixed_configs": str(args.configs) if args.configs else None,
                             "classifier_fit_reused_across_box_configs": True},
                   configurations=configurations, runs=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    baseline_reference = json.loads(args.baseline_results.read_text()) if args.baseline_results else None
    if baseline_reference:
        old_protocol = baseline_reference["protocol"]
        for key in ["classes", "eval_image_ids", "cache_directories"]:
            if old_protocol[key] != protocol[key]:
                parser.error(f"Saved baseline uses a different {key}")
        for seed in seeds:
            if old_protocol["support_image_ids"][str(seed)] != protocol["support_image_ids"][str(seed)]:
                parser.error(f"Saved baseline uses different support images for seed {seed}")
    for seed in seeds:
        ids = protocol["support_image_ids"][str(seed)]
        flame = Flame(model, queries, svm_backend="torch", box_refinement=False,
                      alpha=protocol["alpha"], svm_c=protocol["svm_c"],
                      svm_max_iter=args.svm_max_iter, svm_tol=protocol.get("svm_tol", 1e-3),
                      svm_cache_size_mb=protocol.get("svm_cache_size_mb", 256),
                      svm_working_set_size=protocol.get("svm_working_set_size", 512))
        print(f"Fit Torch FLAME seed={seed}", flush=True)
        synchronize(device)
        start = time.perf_counter()
        flame.fit_support([support_records[i] for i in ids])
        synchronize(device)
        adaptation_seconds = time.perf_counter() - start
        diagnostics = flame.sampling_stats["initial_fit"]["solver"]
        if any(h is None or not h["converged"] for h in diagnostics):
            payload["failed_fit"] = dict(seed=seed, sampling=flame.sampling_stats)
            save()
            raise RuntimeError("Incomplete Torch SVM fit; increase --svm-max-iter before interpreting results")
        print(f"Fitted Torch FLAME seed={seed} in {adaptation_seconds:.3f}s", flush=True)
        heads, head_fits = {}, {}
        for entry in configurations:
            synchronize(device)
            start = time.perf_counter()
            flame.fit_box_refiner([box_records[i] for i in ids], **entry["config"])
            synchronize(device)
            heads[entry["name"]] = flame.box_refiner
            head_fits[entry["name"]] = time.perf_counter() - start
        detections = {name: [] for name in ["baseline", *heads]}
        score_seconds = baseline_post_seconds = 0.0
        box_seconds = defaultdict(float)
        box_post_seconds = defaultdict(float)
        for image in tqdm(eval_images, desc=f"score seed {seed}"):
            embeds, visual, logits, objectness, boxes = (t.to(device) for t in load(image["id"], "eval"))
            synchronize(device)
            start = time.perf_counter()
            with torch.inference_mode():
                scores = flame.scores(embeds, logits, objectness)
            synchronize(device)
            score_seconds += time.perf_counter() - start
            start = time.perf_counter()
            add_detections(detections["baseline"], image["id"], boxes, scores)
            synchronize(device)
            baseline_post_seconds += time.perf_counter() - start
            for name, head in heads.items():
                start = time.perf_counter()
                corrected = head(visual, boxes)
                synchronize(device)
                box_seconds[name] += time.perf_counter() - start
                start = time.perf_counter()
                add_detections(detections[name], image["id"], corrected, scores)
                synchronize(device)
                box_post_seconds[name] += time.perf_counter() - start
        classifier_mb = sum(t.numel() * t.element_size() for h in flame.svms if h
                            for t in h.values() if torch.is_tensor(t)) / 2**20
        for entry in [dict(name="baseline", config=None), *configurations]:
            name = entry["name"]
            result = dict(seed=seed, name=name, config=entry["config"], backend="torch",
                          metrics=metrics(gt, detections[name]), adaptation_seconds=adaptation_seconds,
                          svm_fit_seconds=flame.sampling_stats["initial_fit"]["fit_seconds"],
                          classifier_mb=classifier_mb, classifier_score_seconds=score_seconds,
                          box_fit_seconds=head_fits.get(name, 0), box_score_seconds=box_seconds[name],
                          postprocess_seconds=baseline_post_seconds if name == "baseline" else box_post_seconds[name],
                          box_head_bytes=heads[name].storage_bytes if name in heads else 0,
                          box_support=heads[name].stats if name in heads else None,
                          sampling=flame.sampling_stats, all_fits_converged=True)
            if name == "baseline" and baseline_reference:
                old = next(r for r in baseline_reference["runs"] if r["seed"] == seed and r["name"] == "baseline")
                if old["sampling"]["initial_counts"] != flame.sampling_stats["initial_counts"]:
                    raise RuntimeError("Torch baseline label counts differ from the saved control")
                result["saved_torch_map"] = old["metrics"]["map"]
                result["saved_torch_map_delta"] = result["metrics"]["map"] - old["metrics"]["map"]
            payload["runs"].append(result)
            save()
            print(f"RESULT seed={seed} {name} mAP={result['metrics']['map']:.6f} "
                  f"AP75={result['metrics']['map_75']:.6f} small={result['metrics']['map_small']:.6f}", flush=True)
    if args.selection_output:
        means = {entry["name"]: statistics.mean(r["metrics"]["map"] for r in payload["runs"]
                                                if r["name"] == entry["name"]) for entry in configurations}
        best = max(configurations, key=lambda entry: means[entry["name"]])
        selection = dict(development_results=str(args.output), criterion="highest development mAP among box heads",
                         development_image_paths=sorted(str(p) for p in excluded_paths),
                         development_map=means, configs=[best])
        args.selection_output.parent.mkdir(parents=True, exist_ok=True)
        args.selection_output.write_text(json.dumps(selection, indent=2) + "\n")
        print(f"SELECTED {best['name']} development mAP={means[best['name']]:.6f}", flush=True)


if __name__ == "__main__":
    main()
