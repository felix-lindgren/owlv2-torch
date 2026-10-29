"""Summarize matched, nested sklearn/Nyström FLAME runs and plot fit scaling.

    .venv/bin/python tools/summarize_flame_scaling.py \
      data/flame_features_dior/quick_seed0_1024.json \
      data/flame_features_dior/scaling_{10,20,40}.json \
      --output data/flame_features_dior/scaling_summary
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True, help="Output prefix for JSON, SVG and PNG.")
    args = parser.parse_args()
    inputs = [(path, json.loads(path.read_text())) for path in args.results]
    reference = inputs[0][1]["protocol"]
    shared = ("model", "dataset", "gpu", "annotations", "eval_annotations", "eval_image_ids", "classes",
              "query_template", "support_image_root", "eval_image_root", "alpha", "svm_c", "smoothing")
    points = []
    component_counts, seeds = set(), set()
    zero_shot = inputs[0][1]["zero_shot"]["map"]
    for path, result in inputs:
        protocol = result["protocol"]
        for key in shared:
            if protocol[key] != reference[key]:
                raise ValueError(f"{path}: protocol differs for {key}")
        if abs(result["zero_shot"]["map"] - zero_shot) > 1e-12:
            raise ValueError(f"{path}: zero-shot control differs")
        if len(result["runs"]) != 2 or {r["backend"] for r in result["runs"]} != {"sklearn", "nystrom"}:
            raise ValueError(f"{path}: expected one sklearn and one Nyström run")
        sklearn = next(r for r in result["runs"] if r["backend"] == "sklearn")
        nystrom = next(r for r in result["runs"] if r["backend"] == "nystrom")
        if sklearn["seed"] != nystrom["seed"] or sklearn["support_counts"] != nystrom["support_counts"]:
            raise ValueError(f"{path}: classifier inputs differ")
        seeds.add(sklearn["seed"])
        component_counts.add(nystrom["components"])
        positive = [count["positive"] for count in sklearn["support_counts"]]
        negative = [count["negative"] for count in sklearn["support_counts"]]
        reference_gain = sklearn["metrics"]["map"] - zero_shot
        ids = protocol["support_image_ids"][str(sklearn["seed"])]
        solvers = [s for s in nystrom["solver"] if s]
        points.append({
            "source": str(path), "images_per_class": protocol["support_images_per_class"],
            "support_images": len(ids), "support_image_ids": ids,
            "positive_min": min(positive), "positive_median": statistics.median(positive),
            "positive_max": max(positive), "negative_min": min(negative),
            "negative_median": statistics.median(negative), "negative_max": max(negative),
            "sklearn_fit_seconds": sklearn["fit_seconds"], "nystrom_fit_seconds": nystrom["fit_seconds"],
            "speedup": sklearn["fit_seconds"] / nystrom["fit_seconds"],
            "sklearn_ap": 100 * sklearn["metrics"]["map"], "nystrom_ap": 100 * nystrom["metrics"]["map"],
            "ap_loss": 100 * (sklearn["metrics"]["map"] - nystrom["metrics"]["map"]),
            "gain_retained": ((nystrom["metrics"]["map"] - zero_shot) / reference_gain
                              if reference_gain > 0 else None),
            "sklearn_scoring_seconds": sklearn["score_seconds_including_cache_io"],
            "nystrom_scoring_seconds": nystrom["score_seconds_including_cache_io"],
            "sklearn_classifier_mb": sklearn["classifier_mb"],
            "nystrom_classifier_mb": nystrom["classifier_mb"],
            "nystrom_peak_fit_extra_mb": nystrom["peak_fit_extra_mb"],
            "nystrom_fitted_classes": len(solvers),
            "nystrom_converged_classes": sum(s["converged"] for s in solvers),
            "nystrom_missing_classes": nystrom["missing_classes"],
        })
    if len(seeds) != 1 or len(component_counts) != 1:
        raise ValueError("Support seed and Nyström component count must stay fixed")
    points.sort(key=lambda point: point["support_images"])
    for smaller, larger in zip(points, points[1:]):
        if not set(smaller["support_image_ids"]).issubset(larger["support_image_ids"]):
            raise ValueError("Support image sets are not nested")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {"protocol": {key: reference[key] for key in shared},
               "seed": next(iter(seeds)), "components": next(iter(component_counts)),
               "zero_shot_ap": 100 * zero_shot, "nested_support_sets": True, "points": points}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("| Images/class | Support images | Negatives/class | sklearn (s) | Nyström (s) | Speedup | sklearn AP | Nyström AP |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for p in points:
        print(f"| {p['images_per_class']} | {p['support_images']} | {p['negative_median']:,.0f} | "
              f"{p['sklearn_fit_seconds']:.2f} | {p['nystrom_fit_seconds']:.2f} | {p['speedup']:.2f}× | "
              f"{p['sklearn_ap']:.3f} | {p['nystrom_ap']:.3f} |")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    plt.rcParams["svg.fonttype"] = "none"
    x = [point["negative_median"] / 1000 for point in points]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].plot(x, [p["sklearn_fit_seconds"] for p in points], "o-", color="#2563eb", label="sklearn RBF · CPU")
    gpu_label = reference["gpu"].replace("NVIDIA GeForce ", "")
    axes[0].plot(x, [p["nystrom_fit_seconds"] for p in points], "o-", color="#ea580c", label=f"Nyström · {gpu_label}")
    axes[0].set_yscale("log", base=2)
    axes[0].yaxis.set_major_formatter(ScalarFormatter())
    axes[0].set_ylabel("Fit time for all classes (seconds, log scale)")
    axes[0].set_title("Fitting time")
    axes[0].legend(frameon=False)
    axes[1].plot(x, [p["ap_loss"] for p in points], "o-", color="#7c3aed")
    axes[1].axhline(0, color="#64748b", linestyle="--", linewidth=1)
    axes[1].set_ylim(bottom=min(0, min(p["ap_loss"] for p in points) * 1.1))
    axes[1].set_ylabel("sklearn AP − Nyström AP (points)")
    axes[1].set_title("Accuracy cost of the fixed feature budget")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(x, [f"{value:g}" for value in x])
        axis.set_xlabel("Negatives per class (thousands, log scale)")
        axis.grid(True, which="both", alpha=0.2)
    fig.suptitle(f"{reference['dataset'].upper()}: support-set scaling", fontsize=14)
    fig.text(0.5, 0.02, f"{len(reference['classes'])} classes · {summary['components']} features · "
             f"{len(reference['eval_image_ids'])} fixed evaluation images · one nested support draw · "
             "encoding excluded", ha="center", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(args.output.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(args.output.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(f"Wrote {args.output}.json / .svg / .png")


if __name__ == "__main__":
    main()
