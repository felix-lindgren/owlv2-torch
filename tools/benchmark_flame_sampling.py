"""Compare capped positives and post-fit hard-negative mining using Torch C-SVC.

Reuse an existing cached feature benchmark's exact support/evaluation protocol.
Choose settings with --selection-output on development data, then use --configs
for held-out evaluation. No additional image encoder passes are performed.
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
from OWLv2torch.torch_version.flame_svm import fit_rbf
from benchmark_flame_features import add_detections, metrics, synchronize


def grid():
    result = []
    for cap in [None, 2, 4, 8]:
        result.append(dict(name="baseline" if cap is None else f"cap{cap}",
                           config=dict(max_positives_per_object=cap, mining_rounds=0, mining_nms=None)))
        for threshold in [None, 0.7]:
            suffix = "" if threshold is None else "_diverse"
            result.append(dict(name=f"{'all' if cap is None else f'cap{cap}'}_mined{suffix}",
                               config=dict(max_positives_per_object=cap, mining_rounds=1, mining_nms=threshold)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path)
    parser.add_argument("--configs", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--svm-max-iter", type=int, default=300)
    parser.add_argument("--svm-tol", type=float, default=1e-3)
    parser.add_argument("--svm-cache-size-mb", type=float, default=256)
    parser.add_argument("--svm-working-set-size", type=int, default=512)
    args = parser.parse_args()
    if args.configs and args.selection_output:
        parser.error("Select on development data, then use --configs for held-out evaluation")
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    reference = json.loads(args.reference_results.read_text())
    protocol = reference["protocol"]
    if protocol["model"] != "owlv2-base":
        parser.error("This runner currently supports owlv2-base caches")
    configurations = json.loads(args.configs.read_text())["configs"] if args.configs else grid()
    if not any(entry["name"] == "baseline" for entry in configurations):
        configurations.insert(0, grid()[0])
    seeds = args.seeds or sorted(map(int, protocol["support_image_ids"]))
    full = json.loads(Path(protocol["annotations"]).read_text())
    evaluation = json.loads(Path(protocol["eval_annotations"]).read_text())
    names = protocol["classes"]
    categories = sorted(full["categories"], key=lambda c: c["id"])
    all_names = [c["name"] for c in categories]
    to_query = {c["id"]: names.index(c["name"]) for c in categories if c["name"] in names}
    selected_categories = sorted((c for c in categories if c["id"] in to_query), key=lambda c: to_query[c["id"]])
    query_columns = [all_names.index(name) for name in names]
    eval_ids = set(protocol["eval_image_ids"])
    eval_images = sorted((im for im in evaluation["images"] if im["id"] in eval_ids), key=lambda im: im["id"])
    if len(eval_images) != len(eval_ids) or not eval_images:
        parser.error("Reference evaluation IDs do not match the annotation file")
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
    excluded_paths = {(Path(protocol["eval_image_root"]) / im["file_name"]).resolve()
                      for im in evaluation["images"]}
    for seed in seeds:
        for image_id in protocol["support_image_ids"][str(seed)]:
            if (Path(protocol["support_image_root"]) / images[image_id]["file_name"]).resolve() in excluded_paths:
                parser.error(f"Support/evaluation overlap: seed {seed}, image {image_id}")
    caches = {split: Path(path) for split, path in protocol["cache_directories"].items()}

    def load(image_id, split):
        cached = torch.load(caches[split] / f"{image_id}.pt", weights_only=True)
        if len(cached) == 5:
            embeds, _, logits, objectness, boxes = cached
        else:
            embeds, logits, objectness, boxes = cached
        return embeds, logits[:, query_columns], objectness, boxes

    def query(name):
        return protocol["query_template"].format(name=name.strip().lower().replace("-", " ").replace("_", " "))

    model = OwlV2("base").eval().to(device)
    eval_proposals = [load(im["id"], "eval") for im in tqdm(eval_images, desc="load evaluation cache")]
    support_ids = sorted({image_id for seed in seeds for image_id in protocol["support_image_ids"][str(seed)]})
    support_records = {}
    for image_id in tqdm(support_ids, desc="load support cache"):
        anns = anns_by_image[image_id]
        boxes = [[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in anns)]
        labels = [to_query[a["category_id"]] for a in anns]
        support_records[image_id] = (load(image_id, "support"), boxes, labels)

    # Exclude one-off Triton compilation from repeated adaptation measurements.
    warm_rng = torch.Generator().manual_seed(20261029)
    warm_x = torch.randn(512, 8, generator=warm_rng)
    warm_y = torch.arange(512) % 2
    synchronize(device)
    start = time.perf_counter()
    fit_rbf(warm_x, warm_y, gamma=1 / 8, device=device)
    synchronize(device)
    warmup_seconds = time.perf_counter() - start
    payload = {"protocol": {**protocol, "device": str(device),
                            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                            "svm_backend": "torch", "svm_max_iter": args.svm_max_iter,
                            "svm_tol": args.svm_tol, "svm_cache_size_mb": args.svm_cache_size_mb,
                            "svm_working_set_size": args.svm_working_set_size,
                            "feature_sources": ["class"], "feature_source": "class", "visual_weights": [],
                            "reference_results": str(args.reference_results),
                            "fixed_configs": str(args.configs) if args.configs else None,
                            "evaluated_seeds": seeds, "warmup_seconds": warmup_seconds,
                            "hard_negatives": 50, "random_negatives": 50,
                            "mining_budget": "replace hard negatives; keep random anchors and positives"},
               "configurations": configurations, "runs": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    for seed in seeds:
        support = [support_records[i] for i in protocol["support_image_ids"][str(seed)]]
        for entry in configurations:
            config = entry["config"]
            flame = Flame(model, [query(name) for name in names], svm_backend="torch", box_refinement=False,
                           alpha=protocol["alpha"], svm_c=protocol["svm_c"],
                           svm_max_iter=args.svm_max_iter, svm_tol=args.svm_tol,
                           svm_cache_size_mb=args.svm_cache_size_mb,
                           svm_working_set_size=args.svm_working_set_size)
            print(f"Fit seed={seed} {entry['name']} {config}", flush=True)
            synchronize(device)
            start = time.perf_counter()
            flame.fit_support(support, **config)
            synchronize(device)
            adaptation_seconds = time.perf_counter() - start
            stages = [flame.sampling_stats["initial_fit"], *flame.sampling_stats["rounds"]]
            diagnostics = [s for stage in stages for s in stage["solver"] if s is not None]
            if not diagnostics or not all(s["converged"] for s in diagnostics):
                payload["failed_fit"] = dict(seed=seed, name=entry["name"], sampling=flame.sampling_stats)
                save()
                raise RuntimeError("Unconverged Torch fit; increase --svm-max-iter before interpreting results")
            detections = []
            synchronize(device)
            start = time.perf_counter()
            with torch.inference_mode():
                for image, proposals in zip(eval_images, eval_proposals):
                    features, logits, objectness, boxes = (t.to(device) for t in proposals)
                    scores = flame.scores(features, logits, objectness)
                    add_detections(detections, image["id"], boxes, scores)
            synchronize(device)
            score_seconds = time.perf_counter() - start
            result_metrics = metrics(gt, detections)
            classifier_mb = sum(t.numel() * t.element_size() for h in flame.svms if h
                                for t in h.values() if torch.is_tensor(t)) / 2**20
            result = dict(seed=seed, name=entry["name"], config=config, backend="torch",
                          feature_source="class", adaptation_seconds=adaptation_seconds,
                          fit_seconds=sum(stage["fit_seconds"] for stage in stages),
                          score_seconds=score_seconds, classifier_mb=classifier_mb,
                          sampling=flame.sampling_stats, all_fits_converged=True,
                          missing_classes=[names[c] for c, h in enumerate(flame.svms) if h is None],
                          metrics=result_metrics)
            if entry["name"] == "baseline":
                old = next(r for r in reference["runs"] if r["seed"] == seed and r["backend"] == "sklearn"
                           and r.get("ape_channels") is None and r.get("feature_source", "class") == "class")
                if old["support_counts"] != flame.sampling_stats["initial_counts"]:
                    raise RuntimeError("Baseline label counts differ from the saved control")
                result["cpu_reference_map"] = old["metrics"]["map"]
                result["cpu_reference_map_delta"] = result_metrics["map"] - old["metrics"]["map"]
            payload["runs"].append(result)
            save()
            print(f"RESULT seed={seed} {entry['name']} mAP={result_metrics['map']:.6f} "
                  f"adapt={adaptation_seconds:.3f}s fit={result['fit_seconds']:.3f}s", flush=True)
    if args.selection_output:
        groups = defaultdict(list)
        for row in payload["runs"]:
            if row["name"] != "baseline":
                groups[row["name"]].append(row)
        best = {}
        for rows in groups.values():
            config = rows[0]["config"]
            family = ("combined" if config["mining_rounds"] and config["max_positives_per_object"] is not None
                      else "mining" if config["mining_rounds"] else "capped_positives")
            average = statistics.mean(r["metrics"]["map"] for r in rows)
            if family not in best or average > best[family]["development_map"]:
                best[family] = dict(name=family, config=config, development_map=average,
                                    development_variant=rows[0]["name"])
        selection = dict(development_results=str(args.output), candidate_count=len(configurations) - 1,
                         configs=[grid()[0], *best.values()])
        args.selection_output.parent.mkdir(parents=True, exist_ok=True)
        args.selection_output.write_text(json.dumps(selection, indent=2) + "\n")
        print("SELECTION", json.dumps(selection), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
