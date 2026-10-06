# SFGFusion: Surface Fitting Guided 3D Object Detection with 4D Radar and Camera Fusion

<p align="center">
  <a href="https://www.sciencedirect.com/science/article/pii/S0031320326009647"><img src="https://img.shields.io/badge/Pattern%20Recognition-2026-blue" alt="Pattern Recognition 2026"></a>
  <a href="https://arxiv.org/abs/2510.19215"><img src="https://img.shields.io/badge/arXiv-2510.19215-b31b1b" alt="arXiv"></a>
  <a href="https://doi.org/10.1016/j.patcog.2026.113999"><img src="https://img.shields.io/badge/DOI-10.1016%2Fj.patcog.2026.113999-0a7bbb" alt="DOI"></a>
  <a href="https://lixzzzzzz.github.io/SFGFusion/"><img src="https://img.shields.io/badge/Project-Page-success" alt="Project Page"></a>
</p>

Official implementation of **SFGFusion: Surface fitting guided 3D object detection with 4D radar and camera fusion**, published in *Pattern Recognition* (2026).

**Xiaozhi Li**, Huijun Di, Jian Li, Feng Liu, Wei Liang

[[Paper]](https://www.sciencedirect.com/science/article/pii/S0031320326009647) · [[arXiv]](https://arxiv.org/abs/2510.19215) · [[Project Page]](https://lixzzzzzz.github.io/SFGFusion/)

## 📖 Introduction

SFGFusion is a camera–4D imaging radar fusion framework for 3D object detection. The key idea is to explicitly model object surfaces from image semantics and sparse radar geometry. The fitted surface provides dense object-level depth, which is used in two complementary ways: it guides image-to-BEV view transformation and generates dense surface pseudo-points to compensate for the sparsity of 4D radar point clouds. A separate radar branch preserves the original radar measurements, and the resulting BEV features are fused for final 3D object detection.

The method is evaluated on **TJ4DRadSet** and **View-of-Delft (VoD)**.

## 🛠️ Method

<p align="center">
  <img src="https://arxiv.org/html/2510.19215v1/Model_architecture_overall.png" width="95%" alt="SFGFusion framework">
</p>

SFGFusion contains five main components:

1. **Surface fitting model** — predicts object-wise quadratic surface parameters from image and radar cues.
2. **Surface-fitting-guided image branch** — uses dense fitted depth to improve perspective-view to BEV feature transformation.
3. **4D radar branch** — extracts geometric and motion-aware features from the original radar point cloud.
4. **Surface pseudo-point branch** — projects fitted dense depth into 3D to generate instance-level pseudo-points.
5. **BEV fusion and detection head** — fuses the three BEV representations and predicts 3D bounding boxes and object classes.

## 🔥 Getting Started

This implementation is built on **OpenPCDet v0.6.0** and includes custom modules for 4D radar–camera fusion and surface fitting.

### 1. Environment

We recommend Linux with an NVIDIA GPU and a CUDA-enabled PyTorch installation.

```bash
conda create -n sfgfusion python=3.8 -y
conda activate sfgfusion
```

Install PyTorch and torchvision following the official PyTorch instructions for your CUDA version. Then install `spconv` compatible with your CUDA/PyTorch environment.

Install the Python dependencies and compile the custom CUDA operators:

```bash
pip install -r requirements.txt
python setup.py develop
```

> The codebase follows the OpenPCDet installation mechanism.

### 2. Pretrained image backbone

SFGFusion uses a Swin Transformer image backbone. 

```text
swint-nuimages-pretrained.pth
```

### 3. Dataset Preparation

#### View-of-Delft (VoD)

Download the official [View-of-Delft dataset](https://github.com/tudelft-iv/view-of-delft-dataset) and prepare the accumulated 5-frame radar data in KITTI-style format.

A recommended directory layout is:

```text
data/
└── vod/
    └── radar_5frames/
        ├── ImageSets/
        ├── training/
        │   ├── calib/
        │   ├── image_2/
        │   ├── label_2/
        │   └── velodyne/
        ├── testing/
        ├── kitti_infos_train.pkl
        └── kitti_infos_val.pkl
```

#### TJ4DRadSet

Download the official [TJ4DRadSet](https://github.com/TJRadarLab/TJ4DRadSet) and convert it to the KITTI-style organization expected by the dataloader:

```text
data/
└── TJ4DRadar/
    └── TJ4DRadar/
        ├── ImageSets/
        ├── training/
        │   ├── calib/
        │   ├── image_2/
        │   ├── label_2/
        │   └── velodyne/
        ├── testing/
        ├── kitti_infos_train.pkl
        └── kitti_infos_val.pkl
```

### 4. Instance Masks

The surface fitting module uses precomputed foreground **instance masks** from Mask2Former. Each mask should contain integer instance IDs aligned with the corresponding camera image. The current implementation also uses a frame-to-mask index mapping.

Before training/evaluation, prepare:

```text
data/
└── instance_masks/
    ├── TJ4DRadSet/
    │   ├── index_map.json
    │   └── masks/
    │       ├── mask_0.pt
    │       ├── mask_1.pt
    │       └── ...
    └── VoD/
        └── ...
```

## 🚀 Training

All commands below are run from `tools/`.

### TJ4DRadSet

```bash
cd tools
python train.py --cfg_file cfgs/fusion-TJ4DRadar/2-bevfusion_tj4d_surface_fitting_dual.yaml 
```

### View-of-Delft

```bash
cd tools
python train.py --cfg_file cfgs/kitti_models/1-sfgfusion_vod_dual.yaml
```

## 🧪 Evaluation

### TJ4DRadSet

```bash
cd tools
python test.py \
  --cfg_file cfgs/fusion-TJ4DRadar/2-bevfusion_tj4d_surface_fitting_dual.yaml \
  --ckpt /path/to/sfgfusion_tj4d.pth
```

### View-of-Delft

```bash
cd tools
python test.py \
  --cfg_file cfgs/kitti_models/1-sfgfusion_vod_dual.yaml \
  --ckpt /path/to/sfgfusion_vod.pth
```

## 🙏 Acknowledgement

This project is built upon the excellent open-source [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) toolbox. Parts of the camera-to-BEV implementation are adapted from [BEVFusion](https://github.com/mit-han-lab/bevfusion). We also thank the authors of [TJ4DRadSet](https://github.com/TJRadarLab/TJ4DRadSet) and [View-of-Delft](https://github.com/tudelft-iv/view-of-delft-dataset) for releasing their datasets.

Please follow the licenses and citation requirements of the corresponding upstream projects and datasets.

## ✒️ Citation

If you find SFGFusion useful in your research, please consider citing our paper:

```bibtex
@article{li2026sfgfusion,
  title={SFGFusion: Surface fitting guided 3D object detection with 4D radar and camera fusion},
  author={Li, Xiaozhi and Di, Huijun and Li, Jian and Liu, Feng and Liang, Wei},
  journal={Pattern Recognition},
  volume={180},
  pages={113999},
  year={2026},
  issn={0031-3203},
  doi={10.1016/j.patcog.2026.113999}
}
```

If you use the OpenPCDet codebase, please also cite OpenPCDet as requested by the upstream project.
