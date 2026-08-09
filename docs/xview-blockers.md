# xView pre-training blockers

These are the six issues to resolve before launching the xView training plan in
`docs/xview-run-plan.md`. They are ordered roughly by dependency, not expected
effect size. The aim is to make the first training comparison interpretable;
none of the work below requires a training run.

## Implementation status

All six blockers now have executable paths. The remaining actions are data
preparation/evaluation commands in `docs/xview-run-plan.md`, not code gaps:

1. `tools/compare_object_scales.py` measures the four direct classes after the
   real preprocessing and selects between annotation-only 640 and 960 arms.
2. `docs/xview-eval-spec.json` freezes a disjoint DIOR-20 partition and support
   table; `tools/ovd_eval.py` reports every bucket from the same detections.
3. The external evaluator owns a fixed target vocabulary while checkpoints now
   store their complete resolved training ontology separately.
4. The converter emits uniform and inverse-square-root category-centred streams;
   the trainer enforces a seeded fixed mixture and records actual exposure.
5. Conversion now creates real crops with centre ownership, clipping, padding,
   empty-crop policy and previews. The evaluator loads trainer checkpoints and
   exposes standard versus dense-source `maxDets` explicitly.
6. Training has a fashion-free generic prompt pool plus an aerial profile,
   independent scale augmentation, aerial rotations/flips, and eval-time alias
   pooling into fixed DIOR target classes.

## 1. The proposed tile-scale sweep points in the wrong direction

The plan proposes testing tiles larger than 960 px and downscaling them to the
model input. That makes the classes most useful to DIOR smaller than the model's
16 px patch. A read-only pass over the xView annotations, assigning each box to
one non-overlapping tile by its centre, gave:

| Source crop -> 960 | Vehicle median size | Vehicles below 16 px | Mean boxes/tile | p99 boxes/tile |
|---|---:|---:|---:|---:|
| 640 -> 960 | 19.3 px | 23% | 24 | 249 |
| 960 -> 960 | 12.8 px | 80% | 50 | 499 |
| 1920 -> 960 | 6.4 px | 98% | 166 | 1,448 |

At 1920 -> 960, 80% of road-vehicle boxes are also below 8 px. Nominal dataset
GSD is therefore not a sufficient reason to prefer larger tiles; the relevant
quantity is the post-transform object size for mapped target classes.

### Fix plan

1. Measure the pixel-size distributions of the four directly overlapping DIOR
   classes after DIOR preprocessing.
2. Compare those distributions with xView crops at 640 and 960 px. Do not retain
   1920 px as a primary arm unless the target measurements give a compelling
   reason.
3. Make tile scale explicit converter metadata and hold optimizer steps, prompt
   set, sampling policy and augmentation fixed in any later scale comparison.
4. Consider a controlled 640/960 sampling mixture only after the two individual
   scales are understood.

Ready when: one baseline scale is fixed from measured object sizes, and the
alternative scale changes only field of view and resampling scale.

## 2. The proposed DIOR-supported subset is not actually supported

xView has direct label correspondences for four DIOR classes:

- `Airplane`
- `Ship`
- `Vehicle`
- `Storage tank`

It has no `Harbor` or `Bridge` category. Harbor may benefit indirectly from
ships, cranes and container areas, but that is contextual transfer; bridge is
unseen supervision. Treating all six as directly supported would blur the main
result. The existing base zero-shot AP averaged over the direct four is about
0.229, compared with 0.0706 over all 20 DIOR classes.

### Fix plan

1. Write and freeze an xView-to-DIOR support table before inspecting trained
   results.
2. Report three fixed metric buckets: direct-label overlap, context-related and
   unseen, alongside the full 20-class DIOR mAP.
3. Recompute the stock base and large zero-shot baselines through the final
   checkpoint-evaluation harness for every bucket.
4. Use change in direct-overlap AP as the first transfer signal; use the unseen
   bucket to measure retention or catastrophic forgetting.

Ready when: every DIOR class belongs to exactly one frozen bucket and every
reported trained number has a matched zero-shot baseline.

## 3. Training-ontology changes are being confused with metric changes

Merging xView categories into DIOR-like names changes the supervision and text
queries seen during training. It does not mechanically re-average DIOR mAP if
the DIOR evaluation classes stay fixed. Re-scoring is required when the
evaluation class set changes, not merely because the training ontology changes.

There is also a claim-definition issue. If xView categories are renamed to the
exact DIOR target names, performance on those classes is cross-dataset,
same-vocabulary transfer rather than open-vocabulary transfer.

### Fix plan

1. Keep the DIOR evaluation vocabulary and metric buckets fixed across all
   training-ontology arms.
2. Define two distinct questions: target-aligned transfer for concepts named
   during training, and open-vocabulary retention for DIOR concepts never used
   as training queries.
3. Record the complete xView merge map as run metadata; do not change it after
   seeing results.
4. Precede any merged-training arm with eval-time alias pooling on the existing
   fine-grained checkpoint. If aliases solve the vocabulary mismatch, avoid an
   otherwise redundant training run.

Ready when: a change in DIOR mAP can only come from changed weights or inference
queries, never from silently changing which DIOR classes are averaged.

## 4. Uniform tiling does not align supervision with the target metric

Building and small car comprise about 88% of raw xView boxes. Even after
buildings are removed, directly relevant supervision is still approximately:

| Mapped concept | xView boxes |
|---|---:|
| broad road vehicle | 258,082 |
| ship | 5,141 |
| storage tank | 1,712 |
| airplane | 1,160 |

Road vehicles are therefore about 97% of direct-overlap boxes, while the DIOR
direct-four macro-average gives each concept equal weight. Dropping buildings
alone does not address this mismatch. It is also not neutral: once building
targets are filtered out, visible buildings can contribute as unmatched
objectness background.

### Fix plan

1. Keep a uniform-crop stream so the model continues to see representative
   context and empty regions.
2. Add a category-centred stream sampled with a tempered rule such as inverse
   square-root frequency, then mix it with the uniform stream at a fixed ratio.
3. Decide separately whether building classification should be down-weighted
   while retaining its box/objectness supervision. Do not equate exclusion with
   a class weight of zero.
4. Log crops and retained boxes per mapped superclass, not only raw source-class
   counts, so actual exposure is visible before training starts.

Ready when: the expected crop and positive-box exposure of airplane, ship,
storage tank and vehicle is known, reproducible and no longer an accidental
consequence of the tile grid.

## 5. The conversion and cross-dataset evaluation harnesses are incomplete

The current `tools/convert_xview_to_coco.py` makes a source-image train/holdout
split but writes the original full-size images and boxes; it does not tile. Used
as-is, it would trigger the destructive full-image resize the run plan warns
against.

Cross-dataset validation is also not runnable yet:

- `prototype_train/train_text.py` requires validation category IDs to match the
  training dataset.
- `tools/ovd_eval.py` constructs stock OWLv2 and cannot load a text-training
  checkpoint or select the frozen DIOR metric buckets.
- Raising `--eval-top-k` to 300 does not change the COCO ceiling: both metric
  paths evaluate at most 100 detections per image. Under unique centre
  assignment, about 15.7% of 960 px xView tiles contain more than 100 boxes.

### Fix plan

1. Finish true crop generation: source-level split first, deterministic crop
   coordinates, box remapping, one ownership rule, seam visibility handling,
   edge padding and explicit empty-crop handling.
2. Render representative sparse, dense, edge and seam crops with boxes before
   converting the complete dataset.
3. Add checkpoint loading and independently specified query/class spaces to the
   external evaluator, or add a second validation dataset to the trainer. The
   same path must score stock and fine-tuned weights.
4. Evaluate xView holdout and DIOR separately. Use a clearly documented
   source-side max-detections policy; do not assume `top_k=300` changes standard
   COCO AP.
5. Probe step cost at representative density quantiles, including a dense tail,
   rather than only at the mean.

Ready when: one command can score stock or fine-tuned weights on the frozen DIOR
buckets, and a manual smoke pass confirms that tiled xView boxes still cover the
intended objects.

## 6. Prompt and geometric-augmentation defaults are domain-inappropriate

The text trainer's default prompt pool is `{name}`, `a photo of {name}` and
`a person wearing {name}`. The last template is nonsensical for xView, while the
stored DIOR baseline uses `a satellite photo of {name}`. The trainer also
hardcodes right-angle rotations and vertical flips off even though those
transformations are appropriate for overhead imagery.

The bundled zoom-out and IoU-crop transformations are tied to the broad
`--augment` switch. That can blur a tile-scale comparison by changing effective
object scale inside both arms.

### Fix plan

1. Measure and freeze a small aerial prompt set using stock-weight DIOR
   evaluation, for example `{name}`, `a satellite photo of {name}` and
   `an aerial photo of {name}`. Use the same final evaluation prompt for every
   checkpoint.
2. Remove the fashion-specific template from xView training.
3. Expose and enable 90-degree rotations, horizontal flips and vertical flips
   for xView.
4. Make scale-changing augmentation independently configurable. Disable it for
   the controlled 640-versus-960 comparison, then decide whether to restore it
   for the final recipe.
5. Add eval-time alias pooling for broad DIOR classes such as `Vehicle`, pooling
   fine-grained car, bus and truck queries before scoring the target class.

Ready when: the prompt set, alias map and geometric transforms are explicit in
run metadata and contain no person/fashion-specific defaults.

## Short execution order

1. Freeze the DIOR support buckets and the definition of seen versus unseen.
2. Complete tiled conversion and dual xView/DIOR checkpoint evaluation.
3. Measure target object sizes; choose 640 or 960 and freeze prompts and
   augmentation.
4. Verify crop exposure and introduce the uniform/category-centred sampling mix.
5. Run matched stock-weight baselines and cost probes only after the final
   harness is in place.
6. Use the eventual X1 seed-0 and seed-1 runs as the noise-floor pair rather than
   running a separate duplicate X0d configuration.
