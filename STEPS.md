# STEPS

## Environment and dependencies setup

### SAM 3D Body

#### CUDA 12.4

First install CUDA 12.4 to `/usr/local/cuda-12.4`. Suppose that you are in a Ubuntu 22.04 x86 64 bit system:

```bash
cd /tmp

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb

sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4
```

#### UV environment setup

```bash
uv sync
```

This will setup a `uv` environment for the SAM 3D Body branch.

#### SAM 3D Body repo & weights

Then clone SAM 3D Body's repo under `pkg/`:

```bash
mkdir pkg
cd pkg
git clone org-16943930@github.com:facebookresearch/sam-3d-body.git sam_3d_body
git -C sam_3d_body checkout --detach 3468000660b16c2670503cfe685e6792b36d1ad6
```

_P.S. You don't need to install its dependencies -- we have set them up in our `uv` configs._

Apply for access to model weights in [sam-3d-body-dinov3 | Hugging Face](https://huggingface.co/facebook/sam-3d-body-dinov3). Download it and place the folder under `pkg/sam_3d_body/checkpoints/`. Make sure that `pkg/sam_3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt` and `pkg/sam_3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt` exists.

### AniMer

AniMer uses the separately managed Conda environment named `AniMer`.

```bash
cd pkg
git clone https://github.com/luoxue-star/AniMer.git
cd AniMer
git checkout --detach 4d8be3bd4cdf96183d477c656d2cdbde4509b68e

conda create -n AniMer python=3.10
conda activate AniMer

# install pytorch
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install --force-reinstall "setuptools==81.0.0"
python -m pip install --no-build-isolation "mmcv==1.3.9"

# install repo as a package
python -m pip install --no-build-isolation -e ".[all]"

# install pytorch3d
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

# install pyvista and rtree
python -m pip install pyvista==0.46.4
pip install --only-binary=:all: rtree
```

Downloading the checkpoint folder named AniMer from [here](https://drive.google.com/drive/folders/1xYCJyxZeSYPI6RXnHqa_lDbjnIAs8hVi?usp=sharing) to `data/`. Make sure that `data/AniMer/checkpoints/checkpoint.ckpt` exists.

### PRIMA

```bash
cd pkg
git clone https://github.com/AdaptiveMotorControlLab/PRIMA.git
cd PRIMA
git checkout --detach 0c4251ec8ab03302a2305c02d926d1cf353f9a35

conda create -n prima python=3.10 -y
conda activate prima

PRIMA_PYTHON="$CONDA_PREFIX/bin/python" \
PRIMA_VENV=.venv \
./scripts/clean_install_local.sh

# resolve gradio and huggingface version issue
# and install other packages
source .venv/bin/activate
python -m pip install \
   'gradio>=5.1,<6' \
   'huggingface-hub>=0.34,<1'

python -m pip install pyvista==0.46.4
pip install --only-binary=:all: rtree
```

_P.S. If it raises error about `httpx[socks]`, you are probably under network proxy. Disable it temporarily._

_P.S. The `prima` Conda environment is only used for setup; the runtime environment is `.venv`. Run `source pkg/PRIMA/.venv/bin/activate` from the project root._

## Public datasets

### Public human shape datasets

```bash
uv run python articulate/run.py \
  shape_dir=data/shape-corr/FAUST_r/off \
  out_dir=output/faust_r \
  shape_refiner.trim_mesh=false
```

Benchmarking:

```bash
uv run python match/run.py \
  data.data_root=data/shape-corr/FAUST_r \
  out_dir=output/faust_r
```

#### Uncertainty-aware articulation + soft canonization composition

```bash
uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-corr/FAUST_r/off \
  out_dir=output/faust_r_uncertainty \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_08"

uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-corr/FAUST_r/off \
  out_dir=output/faust_r_uncertainty \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_09"

uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-corr/SCAPE_r/off \
  out_dir=output/scape_r_uncertainty \
  shape_refiner.trim_mesh=false

uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-corr/SHREC19_r/off \
  out_dir=output/shrec19_r_uncertainty \
  shape_refiner.trim_mesh=false

uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-corr/TOPKIDS/off \
  out_dir=output/topkids_uncertainty \
  view_sampler.axis_order=yzx \
  shape_refiner.n_views_surface=0 \
  shape_refiner.trim_mesh=false
```

Benchmarking:

```bash
uv run python match/run.py \
  --config-name partial_aware \
  data.data_root=data/shape-corr/FAUST_r \
  data.articulated_shape_dir=output/faust_r_uncertainty \
  map_extractor.use_reciprocal_consistency=true \
  out_dir=output/faust_r_uncertainty \
  'metrics=[geodist, texture_transfer]' \
  data.return_evecs=true

uv run python match/run.py \
  --config-name partial_aware \
  data=scape_r \
  data.data_root=data/shape-corr/SCAPE_r/ \
  data.articulated_shape_dir=output/scape_r_uncertainty \
  map_extractor.use_reciprocal_consistency=true \
  out_dir=output/scape_r_uncertainty \
  'metrics=[geodist, texture_transfer]' \
  data.return_evecs=true

uv run python match/run.py \
  --config-name partial_aware \
  data=shrec19_r \
  data.data_root=data/shape-corr/SHREC19_r/ \
  data.articulated_shape_dir=output/shrec19_r_uncertainty \
  map_extractor.use_reciprocal_consistency=true \
  out_dir=output/shrec19_r_uncertainty \
  'metrics=[geodist, texture_transfer]' \
  data.return_evecs=true

uv run python match/run.py \
  --config-name partial_aware \
  data=topkids \
  data.data_root=data/shape-corr/TOPKIDS/ \
  data.articulated_shape_dir=output/topkids_uncertainty \
  map_extractor.use_reciprocal_consistency=true \
  out_dir=output/topkids_uncertainty \
  'metrics=[geodist, texture_transfer]' \
  data.return_evecs=true
```

#### Optional spectral-refinement

`spectral-icp`:

```bash
uv run python match/run.py \
  --config-name partial_aware_spectral \
  data.data_root=data/shape-corr/FAUST_r \
  data.articulated_shape_dir=output/faust_r_uncertainty \
  map_extractor.use_reciprocal_consistency=true \
  map_extractor.refinement_method=spectral-icp \
  out_dir=output/faust_r_uncertainty

uv run python match/run.py \
  --config-name partial_aware_spectral \
  data=scape_r \
  data.data_root=data/shape-corr/SCAPE_r/ \
  data.articulated_shape_dir=output/scape_r_uncertainty \
  map_extractor.use_reciprocal_consistency=true \
  map_extractor.refinement_method=spectral-icp \
  out_dir=output/scape_r_uncertainty

uv run python match/run.py \
  --config-name partial_aware_spectral \
  data=shrec19_r \
  data.data_root=data/shape-corr/SHREC19_r/ \
  data.articulated_shape_dir=output/shrec19_r_uncertainty \
  map_extractor.use_reciprocal_consistency=true \
  map_extractor.refinement_method=spectral-icp \
  out_dir=output/shrec19_r_uncertainty

uv run python match/run.py \
  --config-name partial_aware_spectral \
  data=topkids \
  data.data_root=data/shape-corr/TOPKIDS/ \
  data.articulated_shape_dir=output/topkids_uncertainty \
  map_extractor.use_reciprocal_consistency=true \
  map_extractor.refinement_method=spectral-icp \
  out_dir=output/topkids_uncertainty
```

`zoomout`:

```bash
uv run python match/run.py \
  --config-name partial_aware_spectral \
  data.data_root=data/shape-corr/FAUST_r \
  data.articulated_shape_dir=output/faust_r_uncertainty \
  map_extractor.use_reciprocal_consistency=true \
  map_extractor.refinement_method=zoomout \
  out_dir=output/faust_r_uncertainty

uv run python match/run.py \
  --config-name partial_aware_spectral \
  data=scape_r \
  data.data_root=data/shape-corr/SCAPE_r/ \
  data.articulated_shape_dir=output/scape_r_uncertainty \
  map_extractor.use_reciprocal_consistency=true \
  map_extractor.refinement_method=zoomout \
  out_dir=output/scape_r_uncertainty

uv run python match/run.py \
  --config-name partial_aware_spectral \
  data=shrec19_r \
  data.data_root=data/shape-corr/SHREC19_r/ \
  data.articulated_shape_dir=output/shrec19_r_uncertainty \
  map_extractor.use_reciprocal_consistency=true \
  map_extractor.refinement_method=zoomout \
  out_dir=output/shrec19_r_uncertainty

uv run python match/run.py \
  --config-name partial_aware_spectral \
  data=topkids \
  data.data_root=data/shape-corr/TOPKIDS/ \
  data.articulated_shape_dir=output/topkids_uncertainty \
  map_extractor.use_reciprocal_consistency=true \
  map_extractor.refinement_method=zoomout \
  out_dir=output/topkids_uncertainty
```

### Public animal shape datasets

Run the AniMer/SMAL preset from the dedicated Conda environment:

```bash
conda run -n AniMer python articulate/run.py \
  --config-name animer \
  shape_dir=data/shape-corr/SMAL_r/off \
  out_dir=output/smal_r
```

Benchmarking:

```bash
uv run python match/run.py \
  data=smal_r \
  data.data_root=data/shape-corr/SMAL_r \
  data.articulated_shape_dir=output/smal_r \
  out_dir=output/smal_r
```

#### PRIMA as the pose and shape estimator

Run the PRIMA Stage 1/SMAL preset from PRIMA's dedicated virtual environment:

```bash
HF_HUB_OFFLINE=1 pkg/PRIMA/.venv/bin/python -u articulate/run.py \
  --config-name prima \
  shape_dir=data/shape-corr/SMAL_r/off \
  out_dir=output/smal_r_prima
```

Omit `HF_HUB_OFFLINE=1` on the first run so that `imageomics/bioclip` model can be downloaded from Hugging Face and be cached. Detectron2 may also download its detector weights on first use.

Benchmarking:

```bash
uv run python match/run.py \
  data=smal_r \
  data.data_root=data/shape-corr/SMAL_r \
  data.articulated_shape_dir=output/smal_r_prima \
  out_dir=output/smal_r_prima
```

#### Uncertainty-aware articulation + soft canonization composition

Confidence-aware canonization uses the predictor-specific articulation preset and one shared SMAL matching preset:

```bash
conda run -n AniMer python articulate/run.py \
  --config-name uncertainty_aware_animer \
  shape_dir=data/shape-corr/SMAL_r/off \
  out_dir=output/smal_r_uncertainty

HF_HUB_OFFLINE=1 pkg/PRIMA/.venv/bin/python -u articulate/run.py \
  --config-name uncertainty_aware_prima \
  shape_dir=data/shape-corr/SMAL_r/off \
  out_dir=output/smal_r_prima_uncertainty
```

Benchmarking:

```
uv run python match/run.py \
  --config-name partial_aware_smal \
  data=smal_r \
  data.data_root=data/shape-corr/SMAL_r \
  data.articulated_shape_dir=output/smal_r_uncertainty \
  out_dir=output/smal_r_uncertainty

uv run python match/run.py \
  --config-name partial_aware_smal \
  data=smal_r \
  data.data_root=data/shape-corr/SMAL_r \
  data.articulated_shape_dir=output/smal_r_prima_uncertainty \
  out_dir=output/smal_r_prima_uncertainty \
  'metrics=[geodist, texture_transfer]' \
  data.return_evecs=true
```

#### Optional spectral-refinement

`spectral-icp`:

```bash
uv run python match/run.py \
  --config-name partial_aware_smal_spectral \
  data=smal_r \
  data.data_root=data/shape-corr/SMAL_r \
  data.articulated_shape_dir=output/smal_r_uncertainty \
  map_extractor.refinement_method=spectral-icp \
  out_dir=output/smal_r_uncertainty

uv run python match/run.py \
  --config-name partial_aware_smal_spectral \
  data=smal_r \
  data.data_root=data/shape-corr/SMAL_r \
  data.articulated_shape_dir=output/smal_r_prima_uncertainty \
  map_extractor.refinement_method=spectral-icp \
  out_dir=output/smal_r_prima_uncertainty
```

`zoomout`:

```bash
uv run python match/run.py \
  --config-name partial_aware_smal_spectral \
  data=smal_r \
  data.data_root=data/shape-corr/SMAL_r \
  data.articulated_shape_dir=output/smal_r_uncertainty \
  map_extractor.refinement_method=zoomout \
  out_dir=output/smal_r_uncertainty

uv run python match/run.py \
  --config-name partial_aware_smal_spectral \
  data=smal_r \
  data.data_root=data/shape-corr/SMAL_r \
  data.articulated_shape_dir=output/smal_r_prima_uncertainty \
  map_extractor.refinement_method=zoomout \
  out_dir=output/smal_r_prima_uncertainty
```

### In matching fusion

Examples:

```bash
uv run python match/run.py \
  --config-name partial_aware_mhr_refusion \
  data.dataset_name=topkids \
  data.data_root=data/shape-corr/TOPKIDS \
  data.articulated_shape_dir=output/topkids_uncertainty \
  out_dir=output/topkids_uncertainty \
  map_extractor.use_reciprocal_consistency=true \
  data.minimum_iou=0.0 \
  data.use_visibility_in_vertex_confidence=true \
  data.view_residual_weights='{surface: 1, keypoint: 1, iou: 1, inlier: 1}' \
  data.view_temperature=1

conda run -n AniMer python match/run.py \
  --config-name partial_aware_smal_refusion \
  data.data_root=data/shape-corr/SMAL_r \
  data.articulated_shape_dir=output/smal_r_animer_anchor_p2p_no_iou  \
  out_dir=output/smal_r_animer_anchor_p2p_no_iou \
  map_extractor.use_reciprocal_consistency=true \
  data.minimum_iou=0.0 \
  data.use_visibility_in_vertex_confidence=true \
  data.view_residual_weights='{surface: 1000, keypoint: 100, iou: 1, inlier: 1}' \
  data.view_temperature=1e-4

HF_HUB_OFFLINE=1 pkg/PRIMA/.venv/bin/python match/run.py \
  --config-name partial_aware_smal_refusion \
  data.data_root=data/shape-corr/SMAL_r \
  data.articulated_shape_dir=output/smal_r_prima_anchor_p2p_no_iou  \
  out_dir=output/smal_r_prima_anchor_p2p_no_iou \
  map_extractor.use_reciprocal_consistency=true \
  data.minimum_iou=0.0 \
  data.use_visibility_in_vertex_confidence=true \
  data.view_residual_weights='{surface: 1, keypoint: 1, iou: 1, inlier: 1}' \
  data.view_temperature=1
```

## Raw FAUST dataset

### Preprocessing

Download [MPI-FAUST](https://faust-leaderboard.is.tuebingen.mpg.de) dataset to `data/shape-original`.

```bash
cd data/shape-original
unzip MPI-FAUST.zip
mkdir faust_raw
mv MPI-FAUST/training/scans faust_raw/scans
mv MPI-FAUST/training/registrations faust_raw/registrations
rm -rf __MACOSX
rm -rf MPI-FAUST
```

Run the following commands from the repository root. The decimator writes OBJ
meshes and treats the requested vertex count as an approximate target.

#### Decimate the scans

```bash
for resolution in \
  5k:5000 10k:10000 20k:20000 40k:40000 80k:80000 120k:120000
do
  suffix="${resolution%%:*}"
  vertex_count="${resolution##*:}"
  uv run python utils/faust_raw/batch_decimate.py \
    --mesh_dir data/shape-original/faust_raw/scans \
    --mesh_extensions .ply \
    --out_dir "data/shape-original/faust_${suffix}/scans" \
    --decimation "${vertex_count}"
done
```

This creates:

```text
data/shape-original/
├── faust_raw/scans/    # original PLY meshes
├── faust_5k/scans/     # decimated OBJ meshes
├── faust_10k/scans/
├── faust_20k/scans/
├── faust_40k/scans/
├── faust_80k/scans/
└── faust_120k/scans/
```

#### Parse scan-to-template ground truth

The registrations stay in `faust_raw/registrations`; all resolutions use the
same sorted registration meshes as their templates. The exported `s2t/*.txt`
files contain zero-based nearest-registration-vertex indices, one entry per
scan vertex.

```bash
# Full-resolution PLY scans.
uv run python utils/faust_raw/export_s2t.py \
  --raw_mesh_folder data/shape-original/faust_raw/scans \
  --raw_mesh_type ply \
  --regist_mesh_folder data/shape-original/faust_raw/registrations \
  --export_folder data/shape-original/faust_raw

# Decimated OBJ scans.
for suffix in 5k 10k 20k 40k 80k 120k; do
  uv run python utils/faust_raw/export_s2t.py \
    --raw_mesh_folder "data/shape-original/faust_${suffix}/scans" \
    --raw_mesh_type obj \
    --regist_mesh_folder data/shape-original/faust_raw/registrations \
    --export_folder "data/shape-original/faust_${suffix}"
done
```

#### Precompute LBO operators

Use 200 eigenpairs to match the correspondence configurations. The FAUST
preprocessor writes one `spectral/<scan>.npz` per mesh and also writes the
registration-template geodesic matrix to `dist/template_dist.npz`.

```bash
# Full-resolution PLY scans.
uv run python utils/faust_raw/preprocess.py \
  --raw_mesh_folder data/shape-original/faust_raw/scans \
  --raw_mesh_type ply \
  --regist_mesh_folder data/shape-original/faust_raw/registrations \
  --regist_mesh_type ply \
  --output_dir data/shape-original/faust_raw \
  --k_eig 200 \
  --plot_lbo

# Decimated OBJ scans.
for suffix in 5k 10k 20k 40k 80k 120k; do
  uv run python utils/faust_raw/preprocess.py \
    --raw_mesh_folder "data/shape-original/faust_${suffix}/scans" \
    --raw_mesh_type obj \
    --regist_mesh_folder data/shape-original/faust_raw/registrations \
    --regist_mesh_type ply \
    --output_dir "data/shape-original/faust_${suffix}" \
    --k_eig 200 \
    --plot_lbo
done
```

Add `--plot_lbo` to either preprocessing command to also render the first ten
eigenvectors under that dataset's `lbo_vis/` directory.

### Articulation

```bash
# 5k
uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-original/faust_5k/scans \
  out_dir=output/faust_5k \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_08"

uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-original/faust_5k/scans \
  out_dir=output/faust_5k \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_09"

# 10k
uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-original/faust_10k/scans \
  out_dir=output/faust_10k \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_08"

uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-original/faust_10k/scans \
  out_dir=output/faust_10k \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_09"

# 20k
uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-original/faust_20k/scans \
  out_dir=output/faust_20k \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_08"

uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-original/faust_20k/scans \
  out_dir=output/faust_20k \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_09"

# 40k
uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-original/faust_40k/scans \
  out_dir=output/faust_40k \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_08"

uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-original/faust_40k/scans \
  out_dir=output/faust_40k \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_09"

# 80k
uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-original/faust_80k/scans \
  out_dir=output/faust_80k \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_08"

uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-original/faust_80k/scans \
  out_dir=output/faust_80k \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_09"

# 120k
uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-original/faust_120k/scans \
  out_dir=output/faust_120k \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_08"

uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-original/faust_120k/scans \
  out_dir=output/faust_120k \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_09"

# raw
uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-original/faust_raw/scans \
  out_dir=output/faust_raw \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_08"

uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-original/faust_raw/scans \
  out_dir=output/faust_raw \
  shape_refiner.trim_mesh=false \
  shape_loader.filename_filter="_09"
```

### Benchmarking

```bash
uv run python match/run.py \
  --config-name partial_aware_faust_raw_parallel \
  data.data_root=data/shape-original/faust_5k \
  data.articulated_shape_dir=output/faust_5k \
  map_extractor.use_reciprocal_consistency=true \
  out_dir=output/faust_5k

uv run python match/run.py \
  --config-name partial_aware_faust_raw_parallel \
  data.data_root=data/shape-original/faust_10k \
  data.articulated_shape_dir=output/faust_10k \
  map_extractor.use_reciprocal_consistency=true \
  out_dir=output/faust_10k

uv run python match/run.py \
  --config-name partial_aware_faust_raw_parallel \
  data.data_root=data/shape-original/faust_20k \
  data.articulated_shape_dir=output/faust_20k \
  map_extractor.use_reciprocal_consistency=true \
  out_dir=output/faust_20k

uv run python match/run.py \
  --config-name partial_aware_faust_raw_parallel \
  data.data_root=data/shape-original/faust_40k \
  data.articulated_shape_dir=output/faust_40k \
  map_extractor.use_reciprocal_consistency=true \
  out_dir=output/faust_40k

uv run python match/run.py \
  --config-name partial_aware_faust_raw_parallel \
  data.data_root=data/shape-original/faust_80k \
  data.articulated_shape_dir=output/faust_80k \
  map_extractor.use_reciprocal_consistency=true \
  out_dir=output/faust_80k

uv run python match/run.py \
  --config-name partial_aware_faust_raw_parallel \
  data.data_root=data/shape-original/faust_120k \
  data.articulated_shape_dir=output/faust_120k \
  map_extractor.use_reciprocal_consistency=true \
  out_dir=output/faust_120k

uv run python match/run.py \
  --config-name partial_aware_faust_raw_parallel \
  data.data_root=data/shape-original/faust_raw \
  data.raw_mesh_extension=ply \
  data.articulated_shape_dir=output/faust_raw \
  map_extractor.use_reciprocal_consistency=true \
  out_dir=output/faust_raw
```

The preset uses coarse partial-aware matching by default. Enable one of the
optional spectral refinement methods with:

```bash
map_extractor.refinement_method=spectral-icp
map_extractor.refinement_method=zoomout
```

For spectral ICP on high-resolution scans, reduce
`map_extractor.spectral_icp_chunk_size` after an out-of-memory error. The
default is `2048`; smaller chunks reduce peak `torch.cdist` memory at the cost
of additional launches.

### Texture-transfer

```bash
uv run python match/run.py \
  --config-name partial_aware_faust_raw_parallel \
  data.data_root=data/shape-original/faust_5k \
  data.articulated_shape_dir=output/faust_5k \
  map_extractor.use_reciprocal_consistency=true \
  out_dir=output/faust_5k \
  'metrics=[texture_transfer]' \
  data.return_evecs=true \
  batch_start=0 \
  batch_end=39 \
  device=cuda:1

uv run python match/run.py \
  --config-name partial_aware_faust_raw_parallel \
  data.data_root=data/shape-original/faust_10k \
  data.articulated_shape_dir=output/faust_10k \
  map_extractor.use_reciprocal_consistency=true \
  out_dir=output/faust_10k \
  'metrics=[texture_transfer]' \
  data.return_evecs=true \
  batch_start=0 \
  batch_end=39 \
  device=cuda:1

uv run python match/run.py \
  --config-name partial_aware_faust_raw_parallel \
  data.data_root=data/shape-original/faust_20k \
  data.articulated_shape_dir=output/faust_20k \
  map_extractor.use_reciprocal_consistency=true \
  out_dir=output/faust_20k \
  'metrics=[texture_transfer]' \
  data.return_evecs=true \
  batch_start=0 \
  batch_end=39 \
  device=cuda:1

uv run python match/run.py \
  --config-name partial_aware_faust_raw_parallel \
  data.data_root=data/shape-original/faust_40k \
  data.articulated_shape_dir=output/faust_40k \
  map_extractor.use_reciprocal_consistency=true \
  out_dir=output/faust_40k \
  'metrics=[texture_transfer]' \
  data.return_evecs=true \
  batch_start=0 \
  batch_end=39 \
  device=cuda:1

uv run python match/run.py \
  --config-name partial_aware_faust_raw_parallel \
  data.data_root=data/shape-original/faust_80k \
  data.articulated_shape_dir=output/faust_80k \
  map_extractor.use_reciprocal_consistency=true \
  out_dir=output/faust_80k \
  'metrics=[texture_transfer]' \
  data.return_evecs=true \
  batch_start=0 \
  batch_end=39 \
  device=cuda:1

uv run python match/run.py \
  --config-name partial_aware_faust_raw_parallel \
  data.data_root=data/shape-original/faust_120k \
  data.articulated_shape_dir=output/faust_120k \
  map_extractor.use_reciprocal_consistency=true \
  out_dir=output/faust_120k \
  'metrics=[texture_transfer]' \
  data.return_evecs=true \
  batch_start=0 \
  batch_end=39 \
  device=cuda:1

uv run python match/run.py \
  --config-name partial_aware_faust_raw_parallel \
  data.data_root=data/shape-original/faust_raw \
  data.raw_mesh_extension=ply \
  data.articulated_shape_dir=output/faust_raw \
  map_extractor.use_reciprocal_consistency=true \
  out_dir=output/faust_raw \
  'metrics=[texture_transfer]' \
  data.return_evecs=true \
  batch_start=0 \
  batch_end=39 \
  device=cuda:1
```

## BeCoS processed partial dataset

### BeCoS codebase & env

```bash
cd pkg
git clone git@github.com:NafieAmrani/becos-code.git
cd becos-code
git checkout --detach 9fe938335458d92a03c43db7604fc5ef9c7c7570
```

Env setup:

```bash
conda create -n becos python=3.8
conda activate becos
pip install -r requirements.txt
```

### FAUST-only partial-full and partial-partial

Copy configs:

```bash
cp ../../utils/becos/faust.yaml config/
cp ../../utils/becos/faust_partial_full.yaml config/
cp ../../utils/becos/faust_partial_partial.yaml config/
cp ../../utils/becos/shape_dataset.py datasets/
```

Dataset generation:

```bash
ln -sfnT "$(realpath ../../data/shape-original/faust_raw/registrations)" data/faust_regist
mkdir -p ../../data/shape-becos

python main.py --config faust_partial_full
mv output/partial_full ../../data/shape-becos/faust_partial_full

python main.py --config faust_partial_partial
mv output/partial_partial ../../data/shape-becos/faust_partial_partial
```

#### Partial-aware uncertainty articulation

In project root:

```bash
uv run python articulate/run.py \
  --config-name partial_uncertainty_aware_parallel \
  shape_dir=data/shape-becos/faust_partial_full \
  mode=partial_full \
  split=test \
  shape_refiner.trim_mesh=false \
  out_dir=output/faust_becos_p2f

uv run python articulate/run.py \
  --config-name partial_uncertainty_aware_parallel \
  shape_dir=data/shape-becos/faust_partial_partial \
  mode=partial_partial \
  split=test \
  shape_refiner.trim_mesh=false \
  out_dir=output/faust_becos_p2p
```

#### Partial-aware matching and benchmarking

```bash
uv run python match/run.py \
  --config-name partial_aware_becos \
  data.data_root=data/shape-becos/faust_partial_full \
  out_dir=output/faust_becos_p2f

uv run python match/run.py \
  --config-name partial_aware_becos \
  data.data_root=data/shape-becos/faust_partial_partial \
  data.mode=partial_partial \
  map_extractor.matching_mode=partial \
  map_extractor.kappa_factor=1e-3 \
  out_dir=output/faust_becos_p2p
```

The coarse partial-aware result is used by default. Optional spectral
refinement is selected with one of:

```bash
map_extractor.refinement_method=spectral-icp
map_extractor.refinement_method=zoomout
```

### KIDS-only partial-full and partial-partial

The KIDS-only partial-full preset uses `0001.null.0` as its full template.
Unlike the upstream BeCoS metadata, it does not require FAUST as the
partial-full target.

Copy configs:

```bash
cd pkg/becos-code

cp ../../utils/becos/kids.yaml config/
cp ../../utils/becos/kids_partial_full.yaml config/
cp ../../utils/becos/kids_partial_partial.yaml config/
cp ../../utils/becos/templates_kids.csv config/
cp ../../utils/becos/shape_dataset.py datasets/
```

If `data/KIDS/off` is not present, download the KIDS archive:

```bash
python - <<'PY'
from automatic_download import download_dataset

download_dataset(
    "https://cvg.cit.tum.de/_media/spezial/bib/kids.zip",
    "data/KIDS",
    "KIDS",
)
PY
```

Dataset generation:

```bash
python main.py --config kids_partial_full
mv output/partial_full ../../data/shape-becos/kids_partial_full

python main.py --config kids_partial_partial
mv output/partial_partial ../../data/shape-becos/kids_partial_partial
```

#### Partial-aware uncertainty articulation

In project root:

```bash
uv run python articulate/run.py \
  --config-name partial_uncertainty_aware_parallel \
  shape_dir=data/shape-becos/kids_partial_full \
  mode=partial_full \
  split=test \
  shape_refiner.trim_mesh=false \
  out_dir=output/kids_becos_p2f

uv run python articulate/run.py \
  --config-name partial_uncertainty_aware_parallel \
  shape_dir=data/shape-becos/kids_partial_partial \
  mode=partial_partial \
  split=test \
  shape_refiner.trim_mesh=false \
  out_dir=output/kids_becos_p2p
```

#### Partial-aware matching and benchmarking

```bash
uv run python match/run.py \
  --config-name partial_aware_becos \
  data.dataset_name=kids_becos \
  data.data_root=data/shape-becos/kids_partial_full \
  out_dir=output/kids_becos_p2f

uv run python match/run.py \
  --config-name partial_aware_becos \
  data.dataset_name=kids_becos \
  data.data_root=data/shape-becos/kids_partial_partial \
  data.mode=partial_partial \
  map_extractor.matching_mode=partial \
  map_extractor.kappa_factor=1e-3 \
  out_dir=output/kids_becos_p2p
```

The coarse partial-aware result is used by default. Optional spectral
refinement is selected with one of:

```bash
map_extractor.refinement_method=spectral-icp
map_extractor.refinement_method=zoomout
```

## Ablation experiments

The following commands reproduce the three grounding ablations from the legacy
pipeline:

- **w/o grounding** disables both keypoint and surface alignment.
- **w/o kps alignment** disables keypoint alignment only.
- **w/o surf alignment** disables surface alignment only.

All other settings follow the tested articulation and matching presets, including
the dataset-specific overrides, used in the public-shape experiments above.

### TOPKIDS

```bash
# w/o grounding (w/o kps & surf alignment)
uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-corr/TOPKIDS/off \
  out_dir=output/topkids_wo_grounding \
  view_sampler.axis_order=yzx \
  shape_refiner.n_views_surface=0 \
  shape_refiner.trim_mesh=false \
  shape_refiner.optimize_keypoints=false \
  shape_refiner.optimize_surface=false

# w/o kps alignment
uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-corr/TOPKIDS/off \
  out_dir=output/topkids_wo_kps_alignment \
  view_sampler.axis_order=yzx \
  shape_refiner.n_views_surface=0 \
  shape_refiner.trim_mesh=false \
  shape_refiner.optimize_keypoints=false

# w/o surf alignment
uv run python articulate/run.py \
  --config-name uncertainty_aware_parallel \
  shape_dir=data/shape-corr/TOPKIDS/off \
  out_dir=output/topkids_wo_surface_alignment \
  view_sampler.axis_order=yzx \
  shape_refiner.n_views_surface=0 \
  shape_refiner.trim_mesh=false \
  shape_refiner.optimize_surface=false
```

Benchmarking w/ evidence weighted soft map composition:

```bash
for variant in wo_grounding wo_kps_alignment wo_surface_alignment; do
  uv run python match/run.py \
    --config-name partial_aware \
    data=topkids \
    data.data_root=data/shape-corr/TOPKIDS \
    data.articulated_shape_dir="output/topkids_${variant}" \
    map_extractor.use_reciprocal_consistency=true \
    out_dir="output/topkids_${variant}"
done
```

Benchmarking w/ hard-nearest map composition:

```bash
for variant in wo_grounding wo_kps_alignment wo_surface_alignment; do
  uv run python match/run.py \
    data=topkids \
    data.data_root=data/shape-corr/TOPKIDS \
    data.articulated_shape_dir="output/topkids_${variant}" \
    map_extractor.method=spatial-nn \
    out_dir="output/topkids_${variant}"
done
```

### SMAL_r

Run the SMAL ablations with the PRIMA uncertainty-aware preset in PRIMA's
dedicated virtual environment:

```bash
# w/o grounding (w/o kps & surf alignment)
HF_HUB_OFFLINE=1 pkg/PRIMA/.venv/bin/python -u articulate/run.py \
  --config-name uncertainty_aware_prima \
  shape_dir=data/shape-corr/SMAL_r/off \
  out_dir=output/smal_r_prima_wo_grounding \
  shape_refiner.optimize_keypoints=false \
  shape_refiner.optimize_surface=false

# w/o kps alignment
HF_HUB_OFFLINE=1 pkg/PRIMA/.venv/bin/python -u articulate/run.py \
  --config-name uncertainty_aware_prima \
  shape_dir=data/shape-corr/SMAL_r/off \
  out_dir=output/smal_r_prima_wo_kps_alignment \
  shape_refiner.optimize_keypoints=false

# w/o surf alignment
HF_HUB_OFFLINE=1 pkg/PRIMA/.venv/bin/python -u articulate/run.py \
  --config-name uncertainty_aware_prima \
  shape_dir=data/shape-corr/SMAL_r/off \
  out_dir=output/smal_r_prima_wo_surface_alignment \
  shape_refiner.optimize_surface=false
```

Benchmarking w/ evidence weighted soft map composition:

```bash
for variant in wo_grounding wo_kps_alignment wo_surface_alignment; do
  uv run python match/run.py \
    --config-name partial_aware_smal \
    data=smal_r \
    data.data_root=data/shape-corr/SMAL_r \
    data.articulated_shape_dir="output/smal_r_prima_${variant}" \
    out_dir="output/smal_r_prima_${variant}"
done
```

Benchmarking w/ hard-nearest map composition:

```bash
for variant in wo_grounding wo_kps_alignment wo_surface_alignment; do
  uv run python match/run.py \
    data=smal_r \
    data.data_root=data/shape-corr/SMAL_r \
    data.articulated_shape_dir="output/smal_r_prima_${variant}" \
    map_extractor.method=spatial-nn \
    out_dir="output/smal_r_prima_${variant}"
done
```

### FAUST_r

FAUST_r test registrations are `080`–`099`. Run each ablation twice with the
`_08` and `_09` filename filters, writing both batches to the same output
directory:

```bash
# w/o grounding (w/o kps & surf alignment)
for filename_filter in _08 _09; do
  uv run python articulate/run.py \
    --config-name uncertainty_aware_parallel \
    shape_dir=data/shape-corr/FAUST_r/off \
    out_dir=output/faust_r_wo_grounding \
    shape_loader.filename_filter="${filename_filter}" \
    shape_refiner.trim_mesh=false \
    shape_refiner.optimize_keypoints=false \
    shape_refiner.optimize_surface=false
done

# w/o kps alignment
for filename_filter in _08 _09; do
  uv run python articulate/run.py \
    --config-name uncertainty_aware_parallel \
    shape_dir=data/shape-corr/FAUST_r/off \
    out_dir=output/faust_r_wo_kps_alignment \
    shape_loader.filename_filter="${filename_filter}" \
    shape_refiner.trim_mesh=false \
    shape_refiner.optimize_keypoints=false
done

# w/o surf alignment
for filename_filter in _08 _09; do
  uv run python articulate/run.py \
    --config-name uncertainty_aware_parallel \
    shape_dir=data/shape-corr/FAUST_r/off \
    out_dir=output/faust_r_wo_surface_alignment \
    shape_loader.filename_filter="${filename_filter}" \
    shape_refiner.trim_mesh=false \
    shape_refiner.optimize_surface=false
done
```

Benchmarking w/ evidence weighted soft map composition:

```bash
for variant in wo_grounding wo_kps_alignment wo_surface_alignment; do
  uv run python match/run.py \
    --config-name partial_aware \
    data.data_root=data/shape-corr/FAUST_r \
    data.articulated_shape_dir="output/faust_r_${variant}" \
    map_extractor.use_reciprocal_consistency=true \
    out_dir="output/faust_r_${variant}"
done
```

Benchmarking w/ hard-nearest map composition:

```bash
for variant in wo_grounding wo_kps_alignment wo_surface_alignment; do
  uv run python match/run.py \
    data.data_root=data/shape-corr/FAUST_r \
    data.articulated_shape_dir="output/faust_r_${variant}" \
    map_extractor.method=spatial-nn \
    out_dir="output/faust_r_${variant}"
done
```
