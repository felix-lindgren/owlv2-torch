# GPU follow-up: inference speed, 2026-10-28

Follow-up to [the architecture findings](arch-speedup-findings.md), starting
from `d33dd55`. RTX 5070 Ti, 250 W limit, PyTorch 2.11.0+cu130, TensorRT
11.3.0.99, four CPU threads, batch 1, base model, 80 cached COCO query
embeddings. Engines use FP16. Measurements use the padding-free detector.

The useful immediate change is in the evaluator: bypass alias pooling when
each class already corresponds to exactly one logit column. Fixed-resolution
preprocessing and graph replay offer a smaller inference gain. The measured
2:4 sparsity gain is too small to justify prioritizing accuracy-recovery work.

**Applied: bypass identity alias pooling.** `pool_query_logits` previously
indexed and reduced every class separately, including the default configuration
with no aliases. Its 80 singleton reductions cost 2.80 ms in the final isolated
measurement. Returning the input for an identity mapping takes 0.008 ms; actual
alias groups retain the existing reductions.

A manual before/after run of `run_owlv2_inference` on COCO mini500, using blur
0.5, long side 768, multi-label selection, top 300, B=1 and four loader workers:

| pooling | run 1 | run 2 | median | images/s from median |
|---|---:|---:|---:|---:|
| previous | 6.097 s | 5.436 s | 5.767 s | 86.7 |
| identity bypass | 3.802 s | 3.693 s | 3.747 s | 133.4 |

Runs used the order previous/current/current/previous. All four produced the
same SHA-256 of the 150,000 serialized detection records. CPU smoke checks
also matched the previous result for identity, multiple-alias and reordered
mappings. The speedup is **1.54x for this evaluation loop**; the detector
engine itself is unchanged.

**Prototype: compose blur into the separable resize operators.** The current
path resizes a padded square, normalizes, applies a separate Gaussian blur,
then crops. The prototype composes the blur with each cached resize operator,
evaluates just the content rectangle and normalizes last. Padding fill and the
original full-square reflection boundary are included in the composed
operators. The prototype handles RGB uint8 inputs at a fixed preprocessing
configuration.

With source pixels already on the GPU and operators cached, preprocessing
fell from approximately **0.52 to 0.16 ms**. Four real landscape/portrait images
gave maximum normalized-pixel differences of 0.8e-6 to 1.2e-6. These small
differences can change FP16 detections, so accuracy was evaluated separately.

The table below measures decoded PIL image to 300 detections copied back to
the host: image upload, preprocessing, the engine, multi-label selection,
box scaling and host transfer. It excludes image decode, operator construction
and graph capture. Six measurement rounds alternated variant order.

| source image size, H x W | current | composed resize/blur | composed + CUDA graph |
|---|---:|---:|---:|
| 426 x 640 | 4.695 ms | 4.385 ms | 4.149 ms |
| 425 x 640 | 4.714 ms | 4.407 ms | 4.151 ms |
| 480 x 640 | 4.967 ms | 4.679 ms | 4.506 ms |
| 500 x 371 | 4.901 ms | 4.614 ms | 4.379 ms |
| mean of these four images | **4.819 ms** | **4.521 ms** | **4.296 ms** |

That is 1.07x with preprocessing alone and **1.12x combined** over these four
cases. Every captured graph had its own TensorRT execution context and stable
buffers; each timed call uploaded the input and returned host detections.
Separate smoke comparisons of captured versus uncaptured execution on the same
composed pixels produced identical boxes, scores and labels.

Paired COCO accuracy with the same existing engine, B=1, blur 0.5, long side
768, multi-label top 300, standard maxDets 100:

| preprocessing | mini500 mAP | full val5000 mAP |
|---|---:|---:|
| current | 0.514873 | 0.464858 |
| composed | 0.514137 | 0.464613 |
| difference | -0.000736 | **-0.000244** |

The full-val inference loop took 35.87 s versus 35.76 s after the identity
pooling fix. Thus the cached latency gain did **not** translate into a clear
throughput gain for this varied-size image-loading/evaluation workload. The
prototype's 32-entry operator cache had 4,397 hits and 603 misses. Its strongest
current use case is repeated input dimensions, such as a camera stream.

**TensorRT profile and tiling comparison.** All engines came from the same
unmodified ONNX and used builder optimization level 3. Rebuilt engines used
80 queries as their optimization target. Eight rounds alternated engine order.

| engine, input 576 x 768 | median wall latency |
|---|---:|
| original broad profile, optimized at 960 x 960 / 16 queries | 4.276 ms |
| broad profile, optimized at 576 x 768 / 80 queries | 4.177 ms |
| fixed 576 x 768 / 80 queries | 4.100 ms |
| fixed profile, tiling level FULL | 4.098 ms |

Fixed shapes buy about **4%**; FULL tiling adds no clear gain. On one real image,
the fixed engine's top-300 token/class pair overlap with the original was
96.3%, with maximum ranked-score difference 0.0123. It has not received a
dataset accuracy evaluation and remains a scratch benchmark engine.

Capturing only the original engine gave small and inconsistent gains across
the shapes tried. Capturing preprocessing, engine and postprocessing together
gave the more useful fixed-resolution result above.

**Timing-only 2:4 MLP sparsity.** All 24 MLP weight matrices were magnitude
masked to two nonzeros per four values along the MatMul reduction axis. Dense
and sparse engines were then built from these same modified weights, differing
in the `SPARSE_WEIGHTS` flag. These weights were used to measure execution
cost; accuracy recovery was not attempted.

| fixed 576 x 768 engine | median wall latency |
|---|---:|
| original weights, dense | 4.101 ms |
| 2:4 masked weights, dense tactics | 4.086 ms |
| same masked weights, sparse tactics | 3.961 ms |

TensorRT reported 24 eligible MLP layers and selected sparse tactics for all
24. Nevertheless, the matched dense-to-sparse improvement is only **1.03x**.
Selected sparse tactic names indicate FP32 accumulation, whereas the dense
tactics use FP16 accumulation. This result revises the earlier suggestion that
2:4 MLP sparsity might be the best next investment on this GPU/configuration.
The arithmetic estimate of 1.32x assumed a 2x MLP speedup that these kernels
do not deliver.

The default PyTorch cuSPARSELt wrapper was also much slower than its dense
counterpart in an eager timing-only probe. The TensorRT comparison above is
the deployment-relevant result.

**Kernel profile.** At 960 x 960, fused attention consumed 3.823 of 8.897 ms
of summed kernel time (43.0%), and other GEMMs 4.534 ms (51.0%). At 576 x 768,
attention consumed 1.043 of 4.063 ms (25.7%), other GEMMs 2.701 ms (66.5%) and
the rest 0.318 ms (7.8%). Kernel times exclude host gaps and are separate
measurements from the wall-latency tables. MLPs account for approximately 49%
of tower FLOPs at this smaller shape; that number is an arithmetic estimate.

The next model experiment with substantial unmeasured upside remains dense
MLP width reduction followed by reconstruction/distillation, or a learned
selector before the expensive early blocks. The earlier narrower-MLP code
used random replacement weights for timing only; it did not measure accuracy
recovery. Neither experiment was trained in this profiling follow-up.

Raw JSON, logs, prototype scripts and scratch engines are under
`/tmp/owl_gpu_review/`. Main records: `probe_results.json`,
`latency_followup.json`, `engine_comparison.json`,
`preprocessing_accuracy.json`, `preprocessing_accuracy_full.json`,
`sparse_engine_comparison.json` and `eval_pooling.json`. The four-line identity
guard in `tools/ovd_eval.py` is the code change from this follow-up.
