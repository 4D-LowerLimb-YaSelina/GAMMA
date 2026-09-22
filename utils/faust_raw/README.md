## FAUST original dataset

### Full resolution

```bash
# articulation
conda activate sam3d
python -m src.sam3d_grounding.run \
    --mesh_dir /home/knpob/Documents/Hinton/data/shape-original/faust_test/scans \
    --out_dir output/faust_raw \
    --no_trim_mesh \
    --device cuda:1

# parse ground-truth
conda activate fmap
python src/faust_raw/export_s2t.py \
    --raw_mesh_folder /home/knpob/Documents/Hinton/data/shape-original/faust_test/scans \
    --regist_mesh_folder /home/knpob/Documents/Hinton/data/shape-original/faust_test/registrations \
    --export_folder output/faust_raw

# precompute spectral operators
python src/faust_raw/preprocess.py \
    --raw_mesh_folder /home/knpob/Documents/Hinton/data/shape-original/faust_test/scans \
    --regist_mesh_folder /home/knpob/Documents/Hinton/data/shape-original/faust_test/registrations \
    --output_dir output/faust_raw \
    --raw_mesh_type ply \
    --regist_mesh_type ply \
    --k_eig 200 \
    --plot_lbo

# benchmark (mean geodesic distance + texture transfer)
python src/faust_raw/corr-bench.py \
    --data_root output/faust_raw \
    --raw_mesh_folder /home/knpob/Documents/Hinton/data/shape-original/faust_test/scans \
    --regist_mesh_folder output/faust_raw/mhr \
    --mesh_type ply \
    --methods spatial-nn spectral-nn spectral-icp \
    --num_evecs 200 \
    --n_iters 1 \
    --export_texture_transfer \
    --texture_transfer_dir output/faust_raw/texture-transfer \
    --texture_path figures/texture-symbol-grid.png \
    --device cuda
```

### Decimate to 5k vertices

```bash
# articulation
conda activate sam3d
python -m src.sam3d_grounding.run \
    --mesh_dir /home/knpob/Documents/Hinton/data/shape-original/faust_test/scans \
    --out_dir output/faust_raw_5k \
    --no_trim_mesh \
    --decimation 5000 \
    --device cuda:1

# parse ground-truth
conda activate fmap
python src/faust_raw/export_s2t.py \
    --raw_mesh_folder output/faust_raw_5k/decimated \
    --raw_mesh_type obj \
    --regist_mesh_folder /home/knpob/Documents/Hinton/data/shape-original/faust_test/registrations \
    --export_folder output/faust_raw_5k

# precompute spectral operators
python src/faust_raw/preprocess.py \
    --raw_mesh_folder output/faust_raw_5k/decimated \
    --regist_mesh_folder /home/knpob/Documents/Hinton/data/shape-original/faust_test/registrations \
    --raw_mesh_type obj \
    --regist_mesh_type ply \
    --output_dir output/faust_raw_5k \
    --k_eig 200 \
    --plot_lbo

# benchmark (mean geodesic distance + texture transfer)
python src/faust_raw/corr-bench.py \
    --data_root output/faust_raw_5k \
    --raw_mesh_folder output/faust_raw_5k/decimated \
    --regist_mesh_folder output/faust_raw_5k/mhr \
    --mesh_type obj \
    --methods spatial-nn spectral-nn spectral-icp \
    --num_evecs 200 \
    --n_iters 1 \
    --export_texture_transfer \
    --texture_transfer_dir output/faust_raw_5k/texture-transfer \
    --texture_path figures/texture-symbol-grid.png \
    --device cuda
```

## 3dMD jump dataset

```bash
# articulation
conda activate sam3d
python -m src.sam3d_grounding.run \
    --mesh_dir /home/knpob/Documents/Hinton/data/3dMD/S6_nude_jump120_0309_0372/meshes \
    --out_dir output/3dmd_jump \
    --scale_factor 0.001 \
    --device cuda:1

# precompute spectral operators
conda activate fmap
python src/fm/preprocess.py \
    --data_root output/3dmd_jump \
    --mesh_folder trimmed \
    --mesh_type obj \
    --k_eig 200 \
    --no_dist \
    --plot_lbo

# benchmark (texture transfer)
python src/faust_raw/corr-bench.py \
    --data_root output/3dmd_jump \
    --raw_mesh_folder output/3dmd_jump/trimmed \
    --regist_mesh_folder output/3dmd_jump/mhr \
    --mesh_type obj \
    --methods spatial-nn spectral-nn spectral-icp \
    --num_evecs 200 \
    --n_iters 1 \
    --no_geodesic_bench \
    --export_texture_transfer \
    --texture_transfer_dir output/3dmd_jump/texture-transfer \
    --texture_path figures/texture-symbol-grid.png \
    --max_pairs_num 64 \
    --device cuda
```

### Decimate to 5k vertices

```bash
# batch decimation
conda activate sam3d
python src/faust_raw/batch_decimate.py \
    --mesh_dir output/3dmd_jump/trimmed \
    --out_dir output/3dmd_jump_5k/trimmed \
    --decimation 5000

conda activate fmap
python src/fm/preprocess.py \
    --data_root output/3dmd_jump_5k \
    --mesh_folder trimmed \
    --mesh_type obj \
    --k_eig 200 \
    --no_dist \
    --plot_lbo
```

### Decimate to 10k vertices

```bash
# batch decimation
conda activate sam3d
python src/faust_raw/batch_decimate.py \
    --mesh_dir output/3dmd_jump/trimmed \
    --out_dir output/3dmd_jump_10k/trimmed \
    --decimation 10000

conda activate fmap
python src/fm/preprocess.py \
    --data_root output/3dmd_jump_10k \
    --mesh_folder trimmed \
    --mesh_type obj \
    --k_eig 200 \
    --no_dist \
    --plot_lbo
```

### Decimate to 20k vertices

```bash
# batch decimation
conda activate sam3d
python src/faust_raw/batch_decimate.py \
    --mesh_dir output/3dmd_jump/trimmed \
    --out_dir output/3dmd_jump_20k/trimmed \
    --decimation 20000

conda activate fmap
python src/fm/preprocess.py \
    --data_root output/3dmd_jump_20k \
    --mesh_folder trimmed \
    --mesh_type obj \
    --k_eig 200 \
    --no_dist \
    --plot_lbo
```