"""Compare GDA and Tip-Adapter, alone and with FLAME, on an existing proposal cache.

Use an existing benchmark_flame_features.py result as the exact support/eval
protocol. A development sweep writes --selection-output; --configs loads those
fixed settings for held-out evaluation. No image encoder passes are needed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from OWLv2torch import Flame, OwlV2
from OWLv2torch.torch_version.flame_refiners import ProposalRefiner
from benchmark_flame_features import add_detections, metrics, synchronize


def configuration_grid(args):
    configurations = []
    if "gda" in args.methods:
        for shrinkage, weight, svm_weight in itertools.product(
            args.gda_shrinkages, args.gda_weights, (0.0, 1.0)
        ):
            shrinkage = shrinkage if shrinkage == "auto" else float(shrinkage)
            configurations.append(dict(method="gda", gda_shrinkage=shrinkage,
                                       weight=weight, svm_weight=svm_weight))
    if "tip" in args.methods:
        for beta, negative, weight, svm_weight in itertools.product(
            args.tip_betas, (False, True), args.tip_weights, (0.0, 1.0)
        ):
            configurations.append(dict(method="tip", tip_beta=beta, tip_negative=negative,
                                       tip_cache_size=args.tip_cache_size, seed=0,
                                       weight=weight, svm_weight=svm_weight))
    return [{"name": f"{i}_{c['method']}_{'blend' if c['svm_weight'] else 'only'}", "config": c}
            for i, c in enumerate(configurations)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path)
    parser.add_argument("--configs", type=Path, help="Fixed configurations chosen on separate development data.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--methods", nargs="+", choices=["gda", "tip"], default=["gda", "tip"])
    parser.add_argument("--gda-shrinkages", nargs="+", default=["auto", "0.1", "0.5", "0.9"])
    parser.add_argument("--gda-weights", type=float, nargs="+", default=[1.0, 4.0, 16.0])
    parser.add_argument("--tip-betas", type=float, nargs="+", default=[1.0, 5.0, 20.0])
    parser.add_argument("--tip-weights", type=float, nargs="+", default=[8.0, 32.0, 128.0])
    parser.add_argument("--tip-cache-size", type=int, default=128)
    args = parser.parse_args()
    if args.configs and args.selection_output:
        parser.error("Use --selection-output for development and --configs for held-out evaluation")
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    reference = json.loads(args.reference_results.read_text())
    protocol = reference["protocol"]
    if protocol["model"] != "owlv2-base":
        parser.error("This cache runner currently supports owlv2-base")
    configurations = (json.loads(args.configs.read_text())["configs"] if args.configs else configuration_grid(args))
    # Validate all settings before any fitting or scoring.
    grouped = defaultdict(list)
    for entry in configurations:
        config = entry["config"]
        if not (math.isfinite(config["weight"]) and math.isfinite(config["svm_weight"])
                and config["weight"] >= 0 and 0 <= config["svm_weight"] <= 1):
            parser.error("Invalid fusion weights")
        head_options = {k: v for k, v in config.items() if k not in {"weight", "svm_weight"}}
        ProposalRefiner(**head_options)
        grouped[json.dumps(head_options, sort_keys=True)].append(entry)
    seeds = args.seeds or sorted(map(int, protocol["support_image_ids"]))
    names = protocol["classes"]
    full = json.loads(Path(protocol["annotations"]).read_text())
    evaluation = json.loads(Path(protocol["eval_annotations"]).read_text())
    categories = sorted(full["categories"], key=lambda c: c["id"])
    all_names = [c["name"] for c in categories]
    to_query = {c["id"]: names.index(c["name"]) for c in categories if c["name"] in names}
    selected_categories = sorted((c for c in categories if c["id"] in to_query), key=lambda c: to_query[c["id"]])
    query_columns = [all_names.index(name) for name in names]
    eval_ids = set(protocol["eval_image_ids"])
    eval_images = sorted((im for im in evaluation["images"] if im["id"] in eval_ids), key=lambda im: im["id"])
    if len(eval_images) != len(eval_ids) or not eval_images:
        parser.error("Reference evaluation images do not match annotations")
    gt = {**evaluation, "images": eval_images,
          "categories": [{**c, "id": to_query[c["id"]]} for c in selected_categories],
          "annotations": [{**a, "category_id": to_query[a["category_id"]]}
                          for a in evaluation["annotations"]
                          if a["image_id"] in eval_ids and a["category_id"] in to_query]}
    annotations_by_image = defaultdict(list)
    for ann in full["annotations"]:
        if ann["category_id"] in to_query:
            annotations_by_image[ann["image_id"]].append(ann)
    support_images = {im["id"]: im for im in full["images"]}
    eval_paths = {(Path(protocol["eval_image_root"]) / im["file_name"]).resolve() for im in evaluation["images"]}
    for seed in seeds:
        for image_id in protocol["support_image_ids"][str(seed)]:
            if (Path(protocol["support_image_root"]) / support_images[image_id]["file_name"]).resolve() in eval_paths:
                parser.error(f"Support/evaluation overlap for seed {seed}, image {image_id}")
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
    # This historical runner reproduces saved sklearn controls; new sampling
    # experiments use benchmark_flame_sampling.py with the Torch fitter.
    flame = Flame(model, [query(name) for name in names], svm_backend="sklearn", box_refinement=False,
                  alpha=protocol["alpha"], svm_c=protocol["svm_c"])
    proposals = [load(im["id"], "eval") for im in tqdm(eval_images, desc="load evaluation cache")]
    # Keep the large feature arrays on CPU; cache only the small score/box arrays on GPU.
    logits = torch.stack([p[1] for p in proposals]).to(device)
    objectness = torch.stack([p[2] for p in proposals]).to(device)
    boxes = [p[3].to(device) for p in proposals]
    payload = {"protocol": {**protocol, "device": str(device),
                            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                            "reference_results": str(args.reference_results),
                            "fixed_configs": str(args.configs) if args.configs else None,
                            "evaluated_seeds": seeds,
                            "refiner_adaptation": "per-query foreground/background; full normalized embeddings"},
               "configurations": configurations, "runs": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    @torch.inference_mode()
    def evaluate(correction):
        synchronize(device)
        start = time.perf_counter()
        scores = torch.sigmoid(logits + correction) * torch.sigmoid(objectness)[..., None]
        detections = []
        for i, image in enumerate(eval_images):
            add_detections(detections, image["id"], boxes[i], scores[i])
        synchronize(device)
        seconds = time.perf_counter() - start
        return metrics(gt, detections), seconds

    @torch.inference_mode()
    def margins(head):
        synchronize(device)
        start = time.perf_counter()
        values = torch.stack([head.margins(p[0].to(device)) for p in proposals])
        synchronize(device)
        return values, time.perf_counter() - start

    for seed in seeds:
        flame.examples = [[] for _ in names]
        flame._rng = torch.Generator().manual_seed(0)
        for image_id in tqdm(protocol["support_image_ids"][str(seed)], desc=f"support {seed}"):
            anns = annotations_by_image[image_id]
            truth_boxes = [[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in anns)]
            labels = [to_query[a["category_id"]] for a in anns]
            flame._add_proposals(tuple(t.to(device) for t in load(image_id, "support")), truth_boxes, labels)
        synchronize(device)
        start = time.perf_counter()
        flame.fit()
        synchronize(device)
        svm_fit_seconds = time.perf_counter() - start
        base, svm_margin_seconds = margins(flame)
        baseline_metrics, post_seconds = evaluate(flame.alpha * base)
        expected = next(r for r in reference["runs"] if r["seed"] == seed and r["backend"] == "sklearn"
                        and r.get("ape_channels") is None and r.get("feature_source", "class") == "class")
        if abs(expected["metrics"]["map"] - baseline_metrics["map"]) > 1e-6:
            raise ValueError(f"Rebuilt baseline differs from reference for seed {seed}")
        payload["runs"].append(dict(seed=seed, name="flame", config=None, metrics=baseline_metrics,
                                    svm_fit_seconds=svm_fit_seconds, svm_margin_seconds=svm_margin_seconds,
                                    postprocess_seconds=post_seconds))
        save()
        print(f"RESULT seed={seed} FLAME mAP={baseline_metrics['map']:.6f}", flush=True)
        for key, entries in grouped.items():
            options = json.loads(key)
            print(f"Fit refiner seed={seed} {options}", flush=True)
            head = ProposalRefiner(**options)
            synchronize(device)
            start = time.perf_counter()
            head.fit(flame.examples, flame.text.shape[-1], device=device)
            synchronize(device)
            fit_seconds = time.perf_counter() - start
            extra, head_margin_seconds = margins(head)
            for entry in entries:
                config = entry["config"]
                correction = flame.alpha * config["svm_weight"] * base + config["weight"] * extra
                result_metrics, post_seconds = evaluate(correction)
                payload["runs"].append(dict(
                    seed=seed, name=entry["name"], config=config, metrics=result_metrics,
                    svm_fit_seconds=svm_fit_seconds if config["svm_weight"] else 0.0,
                    svm_margin_seconds=svm_margin_seconds if config["svm_weight"] else 0.0,
                    head_fit_seconds=fit_seconds, head_margin_seconds=head_margin_seconds,
                    postprocess_seconds=post_seconds, head_storage_mb=head.storage_bytes / 2**20,
                    support_counts=head.counts, missing_classes=[names[c] for c in head.missing],
                ))
                save()
                print(f"RESULT seed={seed} {entry['name']} mAP={result_metrics['map']:.6f} {config}", flush=True)
            del extra
        del base
    if args.selection_output:
        by_name = defaultdict(list)
        for run in payload["runs"]:
            if run["config"] is not None:
                by_name[run["name"]].append(run)
        best = {}
        for runs in by_name.values():
            config = runs[0]["config"]
            family = f"{config['method']}_{'blend' if config['svm_weight'] else 'only'}"
            average = statistics.mean(r["metrics"]["map"] for r in runs)
            if family not in best or average > best[family]["development_map"]:
                best[family] = dict(name=family, config=config, development_map=average)
        selection = dict(development_results=str(args.output), candidate_count=len(configurations),
                         configs=list(best.values()))
        args.selection_output.parent.mkdir(parents=True, exist_ok=True)
        args.selection_output.write_text(json.dumps(selection, indent=2) + "\n")
        print("SELECTION", json.dumps(selection), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
