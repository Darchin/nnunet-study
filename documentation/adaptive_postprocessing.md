# Training-fitted adaptive post-processing

Adaptive post-processing is disabled by default. When enabled, full-volume validation first predicts the fold's
training cases, fits a policy against their original annotations, and freezes that policy before predicting validation
cases. It does not affect per-epoch validation or cascade next-stage inputs.

Refresh the custom fingerprint before enabling the feature. Extraction recomputes all statistics with standard
progress bars, without partial-statistics checks or migration. Existing preprocessing can be reused:

```powershell
uv run extract_fingerprint -d 123 -fpe PercentileFingerprintExtractor
uv run train 123 CONFIG 0 --postprocess
uv run train 123 CONFIG 0 --val --postprocess --npz
uv run batch_train -d 123 --postprocess
```

Use the existing dataset/configuration/fold and trainer options for your experiment. Training accepts both
`--disable_tta` and `--disable-tta`; batch training uses `--disable-tta`. The same TTA setting is used for training
predictions and validation. Every enabled run fits a fresh policy using the current network, including `--val` runs.
Compatible saved training predictions are reused, so refitting does not necessarily repeat model inference.

CPU post-processing uses a persistent `spawn` process pool. Set its worker count before launching training:

```powershell
$env:nnUNet_n_post_proc = "8"
```

The default is 8; values must be positive integers. This setting is independent of `nnUNet_def_n_proc`, which continues
to control ordinary prediction export and evaluation workers. Export workers are reused for fitting when the counts
match. Workers evaluate competing policies together for each case, reuse immutable morphology/component results,
and send compact metrics to the coordinator. Search decisions and case-order reductions remain deterministic.
Plain phase headers and case progress bars cover cache checking,
export completion, reference verification, candidate preparation, scoring, and report writing. Training prediction keeps
standard per-case logs without a progress bar. Phase headers include region/search metadata where applicable; detailed
trial diagnostics remain in the search report rather than being printed.

## Configuration

Add `post_processing` to a plans configuration to override fitting settings. The block does not enable processing;
`--postprocess` remains necessary. Missing settings receive these defaults:

```json
{
  "post_processing": {
    "hyperparameter_search": "greedy",
    "grouping_percentiles": [10, 25, 50],
    "closing_percentiles": [25, 50, 90],
    "filling_percentiles": [25, 50, 90],
    "size_percentiles": [1, 5, 10],
    "count_percentiles": [90, 95, 100],
    "hierarchy_repair": "auto",
    "aggregation": "case"
  }
}
```

Existing `inherits_from` and `nested_override` rules apply. An explicit operation `null` survives inheritance and disables
that operation; an empty list also disables it. Nonzero signed integers select levels from the ordered catalogue
`[1, 2.5, 5, ..., 97.5, 99, 100]`: `-3` means `[1, 2.5, 5]`, while `3` means `[97.5, 99, 100]`.
Explicit lists select exactly those finite percentiles within `[0,100]`, including levels outside the catalogue.
Levels and resulting numeric candidates are deduplicated; disabled is always included. Explicit lists can exceed the
default three enabled candidates. Invalid settings fail before enabled training begins.

`hierarchy_repair="parent"` trusts parents and clips children; `"child"` trusts children and expands ancestors.
`"auto"` fits both alternatives. `aggregation="instance"` weights every measurement equally; `"case"` gives each case
with measurements equal total weight, divided among its measurements. This affects candidate generation, not the
equally weighted case/region Dice objective.

## Search and geometry

The policy processes independent thresholded region channels before converting them to a label map. Exclusive-label
models use foreground masks from argmax. Containment is derived from dataset label sets. Nested and disjoint regions
are supported; crossing regions, duplicate regions, incompatible output-label encodings, and partially annotated
datasets using an ignore label are rejected when fitting is enabled.

The sequence is hierarchy repair, grouping context, closing, bounded filling, hierarchy consistency, and parent-first
size/count filtering. Grouping does not add voxels: it joins connected components whose shortest physical boundary
voxel-center distance is at most the fitted threshold. Minimum-spanning-tree edges provide the same groups as
thresholding the complete distance graph. Chains of short edges can group objects with widely separated endpoints.
Prediction grouping computes the same threshold-graph membership directly using spatial buckets and bounded boundary
queries, avoiding a complete prediction distance tree. Fingerprints retain exact annotation trees.
Group volume sums original foreground volumes. Prediction groups are reconstructed after morphology and after ancestor
deletion; size and count filters operate on whole groups. Deleting a parent also restricts its descendants.
Removing a child preserves valid parent-only labels.
Morphology additions cannot overwrite existing disjoint regions; dataset order resolves competing additions.

Foreground connectivity is fixed to 26 on volumes and 8 on genuine 2D grids. Remove any `post_processing.connectivity`
key from existing configurations; it is no longer accepted. Background cavities retain full connectivity.
Processing uses original-image
spacing: component and cavity measures are mm³ for volumes and mm² for genuine 2D images; closing radii are mm.
NaturalImage2DIO's singleton axis with spacing 999 is excluded from physical measurements. A 2D model on volumetric
data is still processed in 3D. Closing uses padded, spacing-aware spherical footprints and preserves existing foreground.
It operates within groups of conservatively expanded component bounding boxes, avoiding full-image distance transforms
when objects occupy small areas. This localization preserves the dense operation, including interacting objects and edges.

Defaults propose at most four distinct candidates per operation, including disabled:

| Operation | Enabled candidates |
|---|---|
| Grouping distance | Annotation component-tree distance percentiles 10, 25, 50 |
| Minimum group volume | Training annotation group-volume percentiles 1, 5, 10 |
| Maximum group count | Positive-training-case group-count percentiles 90, 95, 100 |
| Closing radius | Half the fragmentation-gap percentiles 25, 50, 90 |
| Maximum filled cavity volume | Predicted cavity-volume percentiles 25, 50, 90 |

Instance aggregation uses NumPy linear continuous quantiles. Case aggregation sorts observations stably with weights
`1 / measurements_in_case`; interpolation positions are cumulative weights minus half the observation's weight,
normalized so the first and last positions are 0 and 1. A single observation gives a constant threshold. Counts use
inverse empirical-CDF quantiles, with one observation per positive case. Cases without measurements are excluded from
that measurement distribution. No positives disables size/count filtering; no tree distances disables grouping;
no observed fragmentation disables closing; no cavities disables filling.

Component fingerprints retain raw per-case volumes, component IDs and exact distance-tree edges for the applicable
full connectivity only (26 for volumes, 8 for genuine 2D). They also store per-case percentile tables and descriptive
dataset summaries for both aggregation modes. Only raw measurements from the current fold's training identifiers
contribute fitted thresholds; global summaries and validation annotations are never used. For each grouping candidate,
annotation volumes and positive-case counts are regenerated from the stored tree. Refreshing fingerprints recomputes
all measurements without requiring preprocessing again. Metadata contains label-definition fields directly, with no `version` or `spec`
wrapper. Canonical annotation-content digests verify references during fitting without recomputing annotation trees.
Extraction always scans all cases and uses standard progress bars for image statistics and component measurements.
Existing version 1 policies and policies using 6/18-connectivity must be refitted. Compatible version 2 policies using
full connectivity remain usable; explicit 8-connectivity policies apply only to genuine 2D grids.

Requested percentile grids are retained even if they collapse to one value. Reports include duplicate candidate counts
and size thresholds that cannot remove any observed annotation group. No automatic size-threshold substitution occurs.
Trials with provably identical voxel effects share their score: grouping without an effective filter, and size thresholds
at or below the smallest physical voxel volume in the training fold. Requested settings remain present in trial reports.

Fragmentation evidence consists of multiple predicted components each overlapping exactly one common ground-truth
group under the candidate grouping distance. Ambiguous matches are excluded. Shortest physical distances between fragment boundary voxel centers
provide minimum-spanning-tree gap samples. Half-gap radii are proposals; they do not guarantee reconnection.
Cavity candidates depend on the current closing and hierarchy context and are regenerated during search.

For each configured hierarchy direction, the fitter runs parent-first region coordinate sweeps, stopping when there is
no improvement or after five sweeps. Greedy search tries each grouping candidate with sequential downstream fitting
of closing, filling, size, and count. Grouping alone cannot change Dice, so testing its downstream effects is necessary.
Exhaustive search tests joint operation combinations within one region while holding other regions fixed. Default grids
allow up to 1,024 combinations per region/context before deduplication. It is not globally exhaustive across regions.
Every candidate is scored through the complete final pipeline. Greedy search can miss improvements requiring multiple
individually harmful changes; repeated sweeps do not remove that limitation.

The objective is mean per-case Dice averaged equally across evaluation labels/regions, scored after final label-map
conversion. Negative cases participate; two empty masks score 1. Improvements must exceed 1e-8. Ties prefer fewer
enabled operations, then smaller grouping/closing/filling/size thresholds and a larger count cap. The raw prediction wins if
repair offers no improvement. The selected policy can therefore disable every operation.

## Outputs and reuse

Each fold retains:

- `training/`: raw native-grid training predictions and `summary.json`, containing standard raw metrics, selected-policy
  per-case metrics, and aggregate overall/per-region Dice for every distinct tested configuration.
- `training/.postprocessing_cache/`: bit-packed, memory-mappable independent region masks, compact uncompressed
  annotation arrays, geometry, content checksums, and prediction compatibility records.
- `training/postprocessing_search.json`: fold distributions, dependent cavity measurements, tested configurations,
  warnings, worker statistics, and search status. Its `.jsonl` sibling journals completed trials during fitting.
- `validation/`: raw masks, raw `summary.json`, and original probabilities when `--npz` is enabled.
- `validation_postprocessed/`: processed masks and a separate `summary.json`.
- `postprocessing.json`: concrete thresholds, hierarchy policy, connectivity, resolved configuration, quantile conventions,
  candidate grids, objective scores, training identifiers, checkpoint/model identity, label definitions, and TTA setting.
  Raw fold distributions reside in the search report to keep inference/export policy payloads small.

Training predictions and their cache persist after successful fitting. Reuse checks actual model weights, TTA/mirroring,
prediction configuration, input file identities, and output checksums. Missing, incomplete, corrupt, or mismatched cases
are predicted again. Changing only fitting settings reuses predictions. Independent thresholded channels are essential:
an exported region label map alone cannot reconstruct them. Cache files use more disk than compressed masks in return
for fast reads; annotation source/content changes invalidate reference reuse. Older unidentified `.postprocessing-fit-*`
directories are not automatically reused or deleted.

Training summaries are in-sample results. The selected-policy metrics describe applying the policy to the saved raw
predictions; a separate set of processed training masks is not exported. DDP distributes only pending training cases,
keeps heartbeats active while ranks finish uneven assignments, and fits once on rank zero. Both raw and processed
validation metrics are logged. Failed fitting retains reusable completed predictions and journaled trial results.

Ordinary prediction accepts an explicit policy:

```powershell
uv run predict -d 123 -c CONFIG -i INPUT -o OUTPUT --postprocessing-policy PATH_TO_FOLD/postprocessing.json
uv run predict_from_modelfolder -m MODEL -i INPUT -o OUTPUT --postprocessing-policy POLICY.json
uv run ensemble -i PREDICTIONS_A PREDICTIONS_B -o OUTPUT --postprocessing-policy POLICY.json
```

Python callers can pass `postprocessing_policy=path_or_dict` to `nnUNetPredictor` or call
`set_postprocessing_policy(...)` after initialization; `None` disables it. Folder ensembling exposes the same optional
argument. Policies are applied once after probability/logit aggregation; fold policies are never automatically combined.
Saved/returned probabilities stay unmodified. Frozen thresholds/connectivity are independent of the inference
configuration's fitting settings. Policy version and label definitions must match; a differing TTA setting
issues a warning without changing inference settings. An explicitly chosen fold policy for an ensemble has not been
fitted to that ensemble's errors.

Parameters are fitted on **in-sample training predictions**, which may have fewer or different errors than unseen cases.
Validation annotations never contribute to fitting. Training/validation overlap is recorded, and `fold="all"` is labeled
as in-sample evaluation. Revising the search after inspecting validation results is validation-guided development.

## Verification

Run the local tests with `uv run --with pytest python -m pytest nnunetv2/tests`. An isolated GPU smoke test can use an
existing non-cascaded 3D checkpoint with original-grid spacing:

```powershell
uv run python -m nnunetv2.tests.integration_tests.run_adaptive_postprocessing_smoke --model-folder MODEL --fold 0
```

This test reads one training case and one validation case from the real split, takes 64³ crops, fits and exports both
validation variants through actual workers, and removes its temporary outputs. It does not modify the original
fingerprint, preprocessing, or training results. It verifies a small real-data path rather than full-dataset performance.
It also runs validation twice and checks that the second run reuses the training prediction.

Benchmark native-grid preparation and default fitting with one and multiple CPU workers:

```powershell
uv run python -m nnunetv2.tests.integration_tests.benchmark_adaptive_postprocessing -d 226 --fold 0 --cases 4 --processes 1 4 --output benchmark.json
```

The benchmark includes the most fragmented training annotation and uses reproducibly perturbed annotations as synthetic
predictions. It reports wall time, worker memory/time statistics, cache size, and trial/work counts, and checks policy
equality across worker counts. It does not measure model inference or estimate held-out segmentation accuracy.
