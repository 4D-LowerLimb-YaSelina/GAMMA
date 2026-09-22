# Attentive Manifold Feature Refiner + Padded Batched Functional Maps

This repository contains the official implementation of the paper: AMFR: Attentive Manifold Feature Refiner for Unsupervised Non-isometric Shape Matching

> Nonrigid shape correspondence estimation is a fundamental problem in 3D computer vision. DiffusionNet with Functional Maps as its correspondence solver has emerged as a state-of-the-art framework for near-isometric shape correspondence estimation. However, it suffers from the reliance on the low-frequency feature-focused heat diffusion operator and the unstable LBO basis in the presence of topological noise or non-isometric deformation, limiting its application in real-world scenarios. In this work, we propose the Attentive Manifold Feature Refiner (AMFR) to address these limitations, which supports adaptive and flexible long-range shape feature propagation and aggregation. To boost training and inference efficiency, we also proposed a highly efficient padded batched Functional Map (PB-FMap) layer, reducing training time by 3--8 times among different datasets. The proposed method achieved improved accuracy on several challenging shape correspondence benchmarks, especially the non-isometric ones, surpassing even the supervised methods. Code will be released upon acceptance.

## Setup env

```bash
conda create -n fmap python=3.10
conda activate fmap
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

_P.S. If you encouter any issue, you can check the exact package versions listed in [environment.yml](environment.yml)._

```bash
conda env create -f environment.yml
```

## Dataset & preprocessing

To train and test datasets used in this paper, please download the datasets from the this [link](https://drive.google.com/file/d/1zbBs3NjUIBBmVebw38MC1nhu_Tpgn1gr/view?usp=share_link) and put all datasets under `data/`

```bash
├── data
    ├── FAUST_r
    ├── FAUST_a
    ├── SCAPE_r
    ├── SCAPE_a
    ├── SHREC19_r
    ├── TOPKIDS
    ├── SMAL_r
    ├── DT4D_r
    ├── SHREC20
    ├── SHREC16
    ├── SHREC16_test
```

We thank the original dataset providers for their contributions to the shape analysis community, and that all credits should go to the original authors.

### Data preparation
For data preprocessing, we provide *[preprocess.py](preprocess.py)* to compute all things we need. Here is an example for FAUST_r:

```bash
python preprocess.py --data_root data/FAUST_r/ --k_eig 200
```

_P.S. To improve data loading efficiency, the caching mechanism in the original implementation is removed. Folder structure is changed accordingly. Therefore, if you're shifting from the original implementation, you will need to redo the preprocessing with the command above._

_P.S. For the dataset without groundtruth correspondence, you can obmit the geodesic distance matrix calculation with `--no_dist`._

## Checkpoints & configuration & replication

Model weights associated with the reported experiment results are stored under `experiment/<datasets>/<exp-name>/checkpoint/` folder.

Alongside the model weights, all training & benchmarking configurations are stored in `experiment/<datasets>/<exp-name>/config.yaml`.

### Rerun benchmarking

```bash
python bench.py --folder `experiment/<datasets>/<exp-name>/  --device cuda:0
```

To view the results:

```bash
tensorboard --logdir experiment/
```

### Rerun training

```bash
python train.py --config `experiment/<datasets>/<exp-name>/config.yaml` --device cuda:0
```

It will create a new experiment folder under `experiment/<datasets>/<exp-name>-<timestamp>`. To view the results:

```bash
tensorboard --logdir experiment/
```

## Remark

`notebook/prototype` folder contains temporal prototype code during development. It might not be able to run afterwards. Don't use them as a reference.