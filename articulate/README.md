# Articulate Framework

`articulate/` is a Hydra-composable version of the SAM3D-to-MHR articulation workflow. It keeps the original implementation behavior while splitting the pipeline into swappable class components.

## Entry Point

```bash
./.venv/bin/python articulate/run.py \
  shape_loader.shape_dir=/path/to/meshes \
  out_dir=/path/to/output
```

The default configs expect the local `pkg/` layout:

- `pkg/sam_3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt`
- `pkg/sam_3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt`
- `pkg/sam_3d_body/sam_3d_body/...`

You can inspect the composed config without running inference:

```bash
./.venv/bin/python -B articulate/run.py --cfg job
```

## Component Layout

The framework has six Hydra-instantiated components:

- `shape_loader`: loads shapes and applies input-side preprocessing such as scaling and mesh decimation.
- `view_sampler`: samples camera views and renders RGB/depth observations.
- `shape_estimator`: loads SAM3D, estimates MHR parameters, and aligns predictions to the input mesh.
- `shape_prior`: loads the scripted MHR prior, generates meshes/keypoints, and exposes skeleton definitions.
- `shape_refiner`: triangulates joints, optimizes pose/shape, selects views, trims meshes, and merges multi-view predictions.
- `output_handler`: writes meshes, parameters, failed cases, and alignment visualizations.

Each component has a config group under `articulate/config/<component>/`. The default configs point to the current classes through `_target_`.

## Config Structure

General run options live in `articulate/config/run.yaml`, including:

- `out_dir`, `device`

Component-specific options live in their own config groups. For example, input shape settings are in `articulate/config/shape_loader/default.yaml`, view sampling parameters are in `articulate/config/view_sampler/default.yaml`, and optimization parameters are in `articulate/config/shape_refiner/default.yaml`.

Hydra overrides work as usual:

```bash
./.venv/bin/python articulate/run.py \
  shape_loader.shape_dir=/path/to/meshes \
  out_dir=/path/to/output \
  view_sampler.elevation_config=single \
  shape_refiner.n_iters_surface=500 \
  shape_estimator.iou_threshold=0.5
```

## Adding Variants

To add a new implementation of a component:

1. Add a class under the matching `articulate/<component>/` package.
2. Keep the public methods compatible with the existing component role.
3. Put helper functions that are variant-specific in the same component file.
4. Add a new config YAML under `articulate/config/<component>/` with `_target_` pointing to the class.
5. Select it with a Hydra group override, for example `shape_refiner=my_variant`.

Avoid changing shared behavior in the default classes unless the goal is to update the reference implementation.

## Uncertainty-Aware MHR Variant

The confidence-aware SAM 3D Body workflow is available as an isolated preset:

```bash
uv run python articulate/run.py \
  --config-name uncertainty_aware \
  shape_dir=/path/to/meshes \
  out_dir=/path/to/output
```

It keeps the default staged joint and surface optimization, then weights retained
views using their normalized fitting residuals. Template vertices are fused only
from views in which they pass a z-buffer visibility test, with a view-weighted
fallback for vertices that are invisible everywhere. In addition to the default
MHR outputs, it writes `mhr/<mesh_name>.confidence.npz` containing view weights,
vertex visibility support, and cross-view vertex uncertainty.

The performance variant keeps the same output contract while batching ViTDet
and the SAM 3D Body model, overlapping serial EGL rendering with CPU RANSAC,
and overlapping CPU surface sampling with GPU optimization:

```bash
uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=/path/to/meshes \
  out_dir=/path/to/output
```

Failed detector and SAM 3D Body batches fall back to per-view execution.
ViTDet uses bounded GPU batches while background workers prepare its inputs and
perform the subsequent crop transforms. EGL rendering stays on the main thread,
while background alignment workers process a bounded queue of deterministic
RANSAC jobs. Background surface workers maintain a separate bounded queue of
independently seeded Trimesh samples. Relevant overrides are
`shape_estimator.batch_sam3d_inference`,
`shape_estimator.sam3d_inference_batch_size`,
`shape_estimator.batch_view_preparation`,
`shape_estimator.detector_batch_size`,
`shape_estimator.view_preparation_workers`,
`shape_estimator.view_preparation_depth`,
`shape_estimator.alignment_workers`,
`shape_estimator.alignment_prefetch_depth`,
`shape_refiner.surface_sampling_workers`, and
`shape_refiner.surface_prefetch_depth`. The preset uses CUDA-side source
sampling, so its optimization trajectory is not numerically identical to the
serial preset.

## AniMer / SMAL Variant

The AniMer grounding workflow is available as a complete Hydra preset. Run it
from the separately managed `AniMer` Conda environment; the UV environment is
not used or modified by this variant:

```bash
conda run --no-capture-output -n AniMer python -u articulate/run.py \
  --config-name animer \
  shape_dir=/path/to/meshes \
  out_dir=/path/to/output
```

The preset expects a clean AniMer checkout at `pkg/AniMer` and the downloaded
checkpoint folder in its upstream layout:

- `pkg/AniMer/data/AniMer/.hydra/config.yaml`
- `pkg/AniMer/data/AniMer/checkpoints/checkpoint.ckpt`

Detectron2 downloads its Faster R-CNN weights on first use if they are not
already cached. The preset selects the AniMer-specific loader, estimator, SMAL
prior/refiner, and output handler together. Individual groups can also be
selected with `shape_estimator=animer`, `shape_prior=smal`, and corresponding
component overrides, but all AniMer/SMAL groups should be kept paired.

AniMer outputs are written as:

- `smal/<mesh_name>.obj`
- `smal/<mesh_name>.npy`
- `visuals/<mesh_name>_xy.png`
- `visuals/<mesh_name>_zy.png`

The parameter file contains `betas`, `go_aa`, `pose_aa`, and the per-view
alignment values `R`, `t`, and `s`.

Select `--config-name uncertainty_aware_animer` to retain the AniMer lifecycle
while replacing mean fusion with confidence-aware SMAL fusion. In addition to
the files above, this writes `smal/<mesh_name>.confidence.npz` with view
residuals and weights, per-view vertex visibility, visibility support, and
cross-view vertex uncertainty.

## PRIMA / SMAL Variant

The PRIMA Stage 1 workflow is available as a separate Hydra preset. Run it with
the environment created inside the PRIMA checkout; the project UV environment
and the AniMer environment are not used or modified by this variant:

```bash
pkg/PRIMA/.venv/bin/python -u articulate/run.py \
  --config-name prima \
  shape_dir=/path/to/meshes \
  out_dir=/path/to/output
```

The preset expects the clean checkout and Stage 1 assets in their upstream
layout:

- `pkg/PRIMA/data/PRIMAS1/.hydra/config.yaml`
- `pkg/PRIMA/data/PRIMAS1/checkpoints/s1ckpt_inference.ckpt`
- `pkg/PRIMA/data/smal/`

PRIMA also loads `imageomics/bioclip` through Hugging Face and the Detectron2
Faster R-CNN weights used by `demo.py`. Allow network access on the first run.
If BioCLIP is already cached, `HF_HUB_OFFLINE=1` avoids remote metadata checks:

```bash
HF_HUB_OFFLINE=1 pkg/PRIMA/.venv/bin/python -u articulate/run.py \
  --config-name prima \
  shape_dir=/path/to/meshes \
  out_dir=/path/to/output
```

Install the visualization dependency once in the PRIMA environment:

```bash
pkg/PRIMA/.venv/bin/python -m pip install pyvista==0.46.4
```

The PRIMA preset uses the same view sampling, SMAL optimization defaults, and
output schema as AniMer. It suppresses duplicate detections, evaluates retained
animal boxes, selects the aligned candidate with the highest silhouette IoU,
and falls back to the full rendered frame when detection finds no animal.
Outputs are written to `smal/` and `visuals/` with the filenames and parameter
keys documented above for AniMer.

Select `--config-name uncertainty_aware_prima` for the corresponding PRIMA
variant. Its confidence sidecar has the same schema and SMAL vertex ordering as
the AniMer variant, so both outputs can be consumed by the same matcher.

## Outputs

The default output handler writes:

- `mhr/<mesh_name>.obj`
- `mhr/<mesh_name>.npy`
- `trimmed/<mesh_name>.obj` when mesh trimming is enabled
- `decimated/<mesh_name>.obj` when decimation is enabled
- `visuals/<mesh_name>_xy.png`, `visuals/<mesh_name>_xz.png`, `visuals/<mesh_name>_zy.png`
- `failed.txt` for meshes that fail all retry thresholds

The parameter `.npy` stores optimized MHR parameters and selected-view alignment transforms.
The uncertainty-aware preset additionally stores its versioned confidence data in
`mhr/<mesh_name>.confidence.npz` without changing the parameter-file schema.
The uncertainty-aware AniMer and PRIMA presets apply the same compatibility
rule under `smal/`.
