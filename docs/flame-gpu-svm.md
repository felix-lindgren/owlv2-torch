# Full-kernel FLAME SVM on GPU

`Flame(..., svm_backend="torch")` fits the ordinary binary RBF C-SVC on the
model's device and is the default for new `Flame` instances as of 2026-10-29.
Use `svm_backend="sklearn"` for the explicit CPU reference. This backend uses every
labelled proposal, without landmarks, random features, hinge smoothing or a
regularized intercept. It shares FLAME's float16 example storage, float32
standardization, sample standard deviation, `gamma="scale"`, `C` and balanced
class weights with the sklearn reference.

For labels `s` in {-1, +1}, the minimized dual is
`0.5 * (a*s).T @ K @ (a*s) - sum(a)`, subject to
`0 <= a[i] <= C*N/(2*N_class[i])` and `sum(a*s) = 0`.
Each SMO pair update preserves the equality constraint. The intercept is
recovered from free support vectors, or the midpoint of the KKT bounds when
there are no free support vectors.

## Implementation

The solver decomposes the full problem into working sets. Each set contains
the globally most violating pair; later sets retain previous entries to reuse
kernel rows. A second-order pair selection rule solves each local subproblem,
then its coefficient changes update the gradient over **all** examples. This
follows the general [two-level SMO decomposition approach](https://jmlr.org/papers/v22/19-632.html).
Working-set size controls computation and memory, not kernel rank or the
number of examples eligible to become support vectors.

Torch computes full RBF kernel rows and updates the global gradient. A fused
Triton program performs local pair selection and updates on CUDA, avoiding
Python launches and host synchronization for every pair. CUDA installations
without Triton use the Torch inner loop; CPU uses that same reference loop.
The CUDA launch selects the input tensor's device explicitly.

Kernel rows and exported support-vector coefficients use float32. Dual
variables and local gradient arithmetic use float64. Autocast and reduced
float32 matmul precision are disabled while fitting, then the caller's matmul
setting is restored. Before reporting convergence, the solver reconstructs
the full gradient from the support vectors with double-precision accumulation
and checks the global KKT bounds again.

## Controls and diagnostics

```python
flame = Flame(model, queries, svm_backend="torch",
              svm_c=1.0, svm_tol=1e-3, svm_max_iter=300,
              svm_cache_size_mb=256, svm_working_set_size=512)
flame.fit()
diagnostics = [head["fit"] if head else None for head in flame.svms]
```

`svm_tol` bounds the difference between the global upper and lower KKT
bounds, matching sklearn's default tolerance numerically. `svm_max_iter`
counts outer working-set solves; each allows up to
`max(1000, 20 * effective_working_set_size)` pair updates. An iteration-limited
fit retains its current classifier, emits a warning and records
`converged=False`. Increase the iteration budget when this occurs.

The kernel-row allocation stays within `svm_cache_size_mb` MiB. The effective
working set shrinks automatically to fit this budget. Features, optimizer
state, local Gram matrix and construction scratch are additional allocations;
kernel construction and final verification use batches of at most 32 rows.
The budget must accommodate at least two float32 rows. No full N-by-N kernel
matrix is required.

Each fitted head records the outer iterations, pair updates, global KKT gap,
primal and dual objectives, duality gap, equality residual, support-vector
count, effective working-set size, cached kernel memory and kernel evaluations.
`save`/`load` preserve the backend and its controls; loading refits from the
saved examples. Older checkpoints continue to use sklearn.

## Comparing on existing cached proposals

The benchmark accepts `--backends torch`. Components do not apply to this
backend, so it runs once per seed, with a sklearn control on the same examples.
The following repeats the largest existing DIOR scaling point:

```bash
.venv/bin/python tools/benchmark_flame_features.py --dataset-label dior \
  --annotations /mnt/datasets/sat/dior_coco/train/annotations.json \
  --eval-annotations data/flame_features_dior/test_mini200.json \
  --support-image-root /mnt/datasets/sat/dior_coco/train/images \
  --eval-image-root /mnt/datasets/sat/dior_coco/test/images \
  --query-template 'a satellite photo of a {name}' \
  --device cuda:1 --backends torch --seeds 0 --support-images 40 \
  --support-manifest data/flame_features_dior/scaling_support_40.json \
  --cache-dir data/flame_features_dior \
  --output data/flame_features_dior/exact_scaling_40.json
```

To reuse the existing CPU control, add
`--reference-results data/flame_features_dior/scaling_40.json`. This preserves
AP comparisons but does not compute margin differences against the reused
classifier, since the old result file stores metrics rather than fitted heads.
The benchmark reports fit time, peak additional Torch GPU memory, head size,
detection AP and per-class solver diagnostics.

## Initial DIOR measurements

Measured on the RTX 3060 12 GB using the existing nested support manifests,
20 classes and the same held-out 200 evaluation images as the
[approximation scaling experiment](flame-gpu-approximation.md#dior-support-scaling).
AP is on a 0–100 scale. These are single seed-0 measurements with `C=1`,
`alpha=8`, KKT tolerance `1e-3`, a 512-variable working-set limit and a 256 MiB
kernel-row budget. Fit times include standardization, transfers, kernel
construction, optimization and CPU export, and exclude encoding and sampling.

| Images/class | Negatives/class | sklearn fit (s) | Torch GPU fit (s) | Fit speedup | sklearn AP | Torch GPU AP |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 10,000 | 11.33 | 2.11 | 5.38× | 35.9792 | 35.9787 |
| 40 | 79,200 | 596.98 | 15.28 | 39.07× | 40.6124 | 40.6128 |

The small point reruns both fits and compares their margins directly. The
largest point reuses the earlier sklearn control on the identical support
and evaluation split; its speedup therefore compares measurements from
different runs on the same hardware. All 40 GPU classifiers converged after
5–38 outer iterations. The largest final KKT gap was `9.785e-4`, and the
largest equality residual was `7.11e-14`.

At the small point, mean absolute margin difference was `8.88e-5`, maximum
absolute difference `5.64e-4`, with no sign disagreements among the sampled
high-objectness proposals. Additional peak GPU allocation was 50.75 MiB for
the small point and 380.10 MiB for the largest point, above the resident model.
The largest head set occupied 56.96 MiB, close to sklearn's 56.92 MiB.
These observations establish agreement on this split; they are not a general
speed or accuracy guarantee.

Full metrics and per-class diagnostics:
[small point](../data/flame_features_dior/exact_scaling_5.json),
[largest point](../data/flame_features_dior/exact_scaling_40.json).

Manual CPU and CUDA smoke checks also covered nonlinear classification,
duplicate embeddings with conflicting labels, both classes at their box
bounds, a single rare positive, small and odd working sets, and fitting under
autocast/inference mode. Save/load retained the exact backend and its controls;
legacy checkpoints still loaded with sklearn. An intentionally exhausted
iteration budget emitted a warning and recorded non-convergence.
