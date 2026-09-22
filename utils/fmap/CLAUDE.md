# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

AMFR (Attentive Manifold Feature Refiner) — a neural framework for unsupervised non-isometric shape correspondence using DiffusionNet feature extraction and Functional Maps. Includes a Padded Batched Functional Map (PB-FMap) variant that reduces training time 3–8x by vectorizing variable-size mesh batches.

## Environment Setup

```bash
conda create -n fmap python=3.10
conda activate fmap
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Or from the exact lockfile: `conda env create -f environment.yml`

## Key Commands

All commands run from `src/fm/`.

```bash
# 1. Preprocess dataset (compute LBO eigendecomposition + geodesic distance)
python preprocess.py --data_root data/FAUST_r/ --k_eig 200
# For datasets without ground-truth correspondence, skip geodesic distance:
python preprocess.py --data_root data/FAUST_r/ --k_eig 200 --no_dist

# 2. Train
python train.py --config config/fmap/faust.yaml --device cuda:0

# 3. Dry run (quick verification, runs N batches then deletes experiment folder)
python train.py --config config/fmap/faust.yaml --device cuda:0 --dry --drystep 10

# 4. Benchmark (loads best checkpoint from experiment folder)
python bench.py --folder experiment/<dataset>/<exp-name>/ --device cuda:0

# 5. View results
tensorboard --logdir experiment/
```

## Architecture

### Registry-Driven Component System

All components (datasets, networks, losses, metrics, optimizers, schedulers, scripts) are registered via decorators and instantiated by name from YAML config. The registries live in `src/infra/registry.py`:

```python
@NETWORK_REGISTRY.register()
class ULRSSM_PB(nn.Module): ...
```

Config references these by class name (e.g., `network.name: ULRSSM_PB`). The wiring happens in `src/infra/script.py` where `TrainScript` and `BenchScript` look up each component from its registry.

### Config System (YACS)

YAML configs under `config/` define the full experiment: network architecture, sub-modules, datasets, losses, metrics, optimizer, scheduler, and benchmark settings. Loaded via `src/infra/config.py` using `yacs.config.CfgNode`. Config sections are accessed as attributes: `opt.network.name`, `opt.train.epochs`.

### Network Composition Pattern

Networks (in `src/network/`) compose registered modules (in `src/module/`). A typical pipeline:

1. **Point descriptor** (HKS/WKS) → initial per-vertex features
2. **Feature extractor** (DiffusionNet / DiffusionNet_PB) → refined features via learned diffusion
3. **FM solver** (RegularizedFMNet / RegularizedFMNet_PB) → K×K functional map matrix
4. **Permutation matrix** → soft point-to-point correspondence

The `_PB` suffix indicates the padded-batched variant that handles variable-size meshes in a single tensor with masking.

### Training / Benchmarking Pipeline

`TrainScript` in `src/infra/script.py` orchestrates:
- Creates experiment dir `experiment/<exp_name>-<timestamp>/` with config copy, TensorBoard logs, checkpoints
- Runs train loop with configurable test intervals and best-model saving
- `BenchScript` loads `model-best.pth` from an experiment folder and evaluates

`DDPTrainScript` extends this for multi-GPU distributed training via `torch.nn.parallel.DistributedDataParallel`.

### Padded Batch Collation

`src/dataloader/shape_cor_batch.py` contains `collate_shape_batch` — the custom collate function that pads variable-size meshes (vertices, faces, spectral operators, sparse matrices) to the maximum size in the batch and produces vertex masks. This is critical for batched computation with variable-topology meshes.

### Data Layout After Preprocessing

```
data_root/
├── off/            # or obj/, ply/ — original meshes
├── spectral/       # LBO eigendecomposition (.npz per mesh)
├── dist/           # geodesic distance matrices (.npz per mesh)
└── corr/           # ground-truth correspondences (if available)
```

`preprocess.py` populates `spectral/` and `dist/`. The `corr/` directory comes with the downloaded datasets.

### Functional Map Utilities

- `src/utils/fmap.py`: `fmap2pointmap` converts K×K functional map to vertex correspondence via nearest-neighbor in spectral space; `pointmap2fmap` does the inverse
- `src/utils/geometry.py`: `compute_operators` computes LBO, gradient operators, mass matrix; `to_basis`/`from_basis` project between spatial and spectral domains

## Adding New Components

1. Create a class in the appropriate `src/` subdirectory
2. Decorate with `@<REGISTRY>.register()` (this auto-registers using the class name)
3. Reference the class name in your YAML config

No imports needed in a central location — the registry decorator handles discovery, but the module must be imported somewhere (the existing `__init__.py` files handle this via wildcard or explicit imports).
