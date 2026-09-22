# Post-hoc AniMer/PRIMA-SMAL and MHR refusion

The refusion match presets rebuild the optimized mesh for each saved view,
filter views by their stored IoU, and repeat confidence-weighted fusion in
memory. `partial_aware_smal_refusion` retains the PRIMA behavior,
`partial_aware_smal_refusion_animer` loads AniMer-SMAL artifacts, and
`partial_aware_mhr_refusion` loads SAM 3D Body/MHR artifacts. This makes fusion
parameters cheap to retune without repeating articulation optimization.

Post-hoc refusion does not rerun optimization. Always use artifacts from an
articulation run with IoU threshold `0.0` and `n_views_surface: 0`; otherwise
rejected or truncated views are absent and cannot be recovered. Changing the
visibility depth tolerance or pixel radius is also out of scope because
refusion reuses the saved visibility masks.

## Required artifacts and runtime

For every benchmark shape `<shape>`, an SMAL artifact root must contain:

```text
<artifact-root>/
└── smal/
    ├── <shape>.npy
    └── <shape>.confidence.npz
```

The SMAL parameter file supplies `betas`, `go_aa`, `pose_aa`, `R`, `t`, and `s`
for every optimized view. An MHR artifact root instead contains
`mhr/<shape>.npy`, `mhr/<shape>.confidence.npz`, and the saved
`mhr/<shape>.obj`; its parameter file supplies `identity_coeffs`,
`model_parameters`, `face_expr_coeffs`, `R`, `t`, and `s`. MHR refusion uses
the OBJ only for its fixed face topology and validates it against the scripted
model's vertex count.

The `.confidence.npz` sidecar is mandatory for every backend. Ordinary
articulation outputs without it are unsupported because they do not contain
the named per-view residual terms or raw vertex visibility masks required for
refusion.

### PRIMA-SMAL

Refusion must use the same PRIMA checkout, checkpoint, and SMAL assets that
produced the artifacts.

Run matching in PRIMA's environment so its checkpoint dependencies are
available. Detectron2 is not initialized by refusion. Omit `HF_HUB_OFFLINE=1`
on the first run if PRIMA's BioCLIP dependency is not already cached.

```bash
HF_HUB_OFFLINE=1 pkg/PRIMA/.venv/bin/python match/run.py \
  --config-name partial_aware_smal_refusion \
  data.data_root=/path/to/SMAL_r \
  data.articulated_shape_dir=/path/to/iou_0_articulation_output \
  out_dir=output/smal_refusion_baseline \
  metrics.geodist.export_path=output/smal_refusion_baseline/geodist.csv
```

`data.artifact_root` defaults to `data.articulated_shape_dir`. Override
`data.prima_pkg_path` and `data.checkpoint_path` when the PRIMA checkout or
checkpoint is elsewhere. Results remain in memory; the run does not write
alternate OBJ or confidence artifacts. Set a distinct `out_dir` and metric
export path for each experiment so results are not overwritten.

### AniMer-SMAL

AniMer refusion loads the customized AniMer checkpoint loader and retains only
`model.smal`; Detectron2 is not imported or initialized. After activating an
environment containing AniMer's checkpoint dependencies:

```bash
python match/run.py \
  --config-name partial_aware_smal_refusion_animer \
  data.data_root=/path/to/SMAL_r \
  data.articulated_shape_dir=/path/to/animer_iou_0_output \
  out_dir=output/animer_smal_refusion
```

Override `data.animer_pkg_path` and `data.checkpoint_path` when those assets are
elsewhere.

### SAM 3D Body/MHR

MHR refusion loads only the scripted `mhr_model.pt`; it does not load SAM 3D
Body inference, a detector, or an FOV model. The preset defaults to FAUST:

```bash
uv run python match/run.py \
  --config-name partial_aware_mhr_refusion \
  data.data_root=/path/to/FAUST_r \
  data.articulated_shape_dir=/path/to/mhr_iou_0_output \
  out_dir=output/faust_mhr_refusion
```

The existing human datasets are supported with paired overrides. For example,
use `data.dataset_name=scape_r data.data_root=/path/to/SCAPE_r` (and the
corresponding `data.articulated_shape_dir`) for SCAPE. The other accepted names
are `shrec19_r` and `topkids`. Override `data.mhr_path` to select a different
scripted MHR asset.

## Fusion calculation

For each surviving view $v$, the saved residual terms are combined as

$$
r_v = \lambda_s q_{v,s} + \lambda_k q_{v,k}
    + \lambda_i q_{v,i} + \lambda_n q_{v,n},
$$

where the terms are surface, keypoint, IoU penalty, and inlier penalty. The IoU
and inlier terms are stored as $q_{v,i}=1-\operatorname{IoU}_v$ and
$q_{v,n}=1-\operatorname{inlierRatio}_v$. View weights use a negative softmax:

$$
\alpha_v = \frac{\exp(-r_v/T)}{\sum_u \exp(-r_u/T)}.
$$

For vertex $j$, only views whose saved visibility mask $m_{v,j}$ is true
normally contribute:

$$
\bar{x}_j =
\frac{\sum_v \alpha_v m_{v,j} x_{v,j}}
     {\max(\sum_v \alpha_v m_{v,j}, \epsilon)}.
$$

If no selected view sees a vertex, fusion falls back to all selected views for
that vertex. The resulting uncertainty is the analogously weighted mean
squared deviation from $\bar{x}_j$.

`data.use_visibility_in_vertex_confidence` controls only the fused geometry and
its vertex uncertainty. Its default, `true`, uses $\alpha_v m_{v,j}$ exactly as
above. With `false`, the fusion mask is replaced by all ones, so every surviving
view contributes to every vertex. In both modes the in-memory diagnostics keep
the raw saved `vertex_visibility`, and `vertex_visibility_support` plus
`vertex_visibility_fallback` are still computed from the raw masks and view
weights. Therefore the independent downstream
`map_extractor.use_visibility_term` behavior is unchanged.

Disabling visibility can stabilize very sparse selected-view sets and avoids
per-vertex fallback discontinuities, but it also lets occluded or back-facing
view estimates influence geometry. Keeping visibility enabled preserves
complementary visible-surface evidence, although aggressive IoU filtering can
leave low support for large regions. Treat the flag and minimum-IoU threshold
as separate ablations.

## Tunable parameters

| Hydra parameter | Effect and trade-off |
| --- | --- |
| `data.minimum_iou` | Hard, inclusive gate on the original IoU recovered as `1 - iou_penalty`. Raising it removes poorly aligned silhouettes, but reduces view diversity and can discard useful geometry. The run fails clearly if a shape has no surviving view. |
| `data.view_residual_weights.surface` | Weight on scale-normalized bidirectional surface Chamfer. Larger values favor views closest to the target surface, but can make fusion depend on very few similar views and lose complementary coverage. |
| `data.view_residual_weights.keypoint` | Weight on scale-normalized keypoint error. It can reject bad poses, but predicted or triangulated joints may be biased; over-weighting it can select a worse surface fit. |
| `data.view_residual_weights.iou` | Weight on `1 - IoU`. It favors silhouette agreement, but IoU is sensitive to masks, camera alignment, occlusion, and foreshortening. Use the hard gate for clearly unacceptable views and this weight for softer ranking. |
| `data.view_residual_weights.inlier` | Weight on `1 - inlier_ratio` from robust alignment. It favors well-supported alignments, although sparse alignment support is not always a reliable proxy for dense surface accuracy. |
| `data.view_temperature` | Softmax temperature $T$. Lower values concentrate weight on the best residual and may improve accuracy when ranking is reliable, but increase winner-take-all behavior and sensitivity to noisy terms. Higher values approach uniform fusion and preserve diversity, but retain more poor views. Must be positive. |
| `data.fusion_eps` | Numerical support threshold and denominator guard. Keep the default `1e-8` for accuracy experiments; changing it generally matters only for near-zero visibility support and is primarily a stability control. Must be positive. |
| `data.use_visibility_in_vertex_confidence` | `true` uses saved visibility for fused geometry and uncertainty. `false` gives every surviving view full per-vertex fusion support, while retaining raw visibility support/fallback diagnostics for matching. |

A residual weight of zero disables that term. The saved `alignment` residual is
retained in the artifact but is not part of the post-hoc weighted total.

Only the ratios $\lambda/T$ affect the softmax. Multiplying every residual
weight by a constant $c$ has the same effect as dividing the temperature by
$c$, so do not treat global weight scale and temperature as independent tuning
dimensions. Relative weights still matter because the residual terms have
different numerical scales.

## Suggested tuning workflow

1. Run the exact baseline (`minimum_iou=0`, all weights `1`, temperature `1`)
   and retain its per-pair geodesic CSV.
2. Sweep `minimum_iou` first, for example `0.0`, `0.4`, `0.5`, `0.6`, and `0.7`.
   Check that every shape retains at least one view.
3. Fix the best gate and tune relative residual weights. Inspect the saved term
   magnitudes before choosing a grid: surface and keypoint terms are normalized
   geometric errors and can be much smaller than the dimensionless IoU and
   inlier penalties.
4. Tune temperature after the relative ranking is useful. A compact first grid
   is `0.25`, `0.5`, `1.0`, and `2.0`.
5. Re-evaluate the best candidates with the same dataset split, map-extractor
   settings, and checkpoint. Change only fusion parameters when attributing an
   accuracy difference to refusion.

A simple one-dimensional IoU sweep can be run as separate, reproducible jobs:

```bash
for MIN_IOU in 0.0 0.4 0.5 0.6 0.7; do
  RUN_DIR="output/smal_refusion_iou_${MIN_IOU}"
  HF_HUB_OFFLINE=1 pkg/PRIMA/.venv/bin/python match/run.py \
    --config-name partial_aware_smal_refusion \
    data.data_root=/path/to/SMAL_r \
    data.articulated_shape_dir=/path/to/iou_0_articulation_output \
    data.minimum_iou="$MIN_IOU" \
    out_dir="$RUN_DIR" \
    metrics.geodist.export_path="$RUN_DIR/geodist.csv"
done
```

Each job reloads the checkpoint, but reconstructs every unique shape only once
and reuses it across all pairs within that job.

The intended first non-baseline experiment is:

```bash
RUN_DIR=output/smal_refusion_iou060_w1000_100_t050
HF_HUB_OFFLINE=1 pkg/PRIMA/.venv/bin/python match/run.py \
  --config-name partial_aware_smal_refusion \
  data.data_root=/path/to/SMAL_r \
  data.articulated_shape_dir=/path/to/iou_0_articulation_output \
  data.minimum_iou=0.6 \
  data.view_residual_weights.surface=1000 \
  data.view_residual_weights.keypoint=100 \
  data.view_residual_weights.iou=1 \
  data.view_residual_weights.inlier=1 \
  data.view_temperature=0.5 \
  out_dir="$RUN_DIR" \
  metrics.geodist.export_path="$RUN_DIR/geodist.csv"
```

Each match run loads its model once and reconstructs each unique shape once;
fused vertices, faces, and confidence are reused across repeated dataset pairs.
Start a new run for every parameter setting. Mutating fusion parameters after a
shape has entered the cache would otherwise leave a stale fused result.
