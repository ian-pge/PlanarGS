# PlanarGS: High-Fidelity Indoor 3D Gaussian Splatting Guided by Vision-Language Planar Priors

This repository contains the implementation of **PlanarGS**.

## Installation & Usage (Simplified with Pixi)

This project is now managed with `pixi` for easy reproducibility.

### 1. Install Pixi
If you don't have pixi installed, run:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 2. Setup & Download
Run the following commands to install dependencies, download the dataset, and fetch necessary checkpoints.
```bash
# Clone the repository
git clone --recursive https://github.com/YourUsername/PlanarGS.git
cd PlanarGS

# Download Dataset (Replica/room0) and Checkpoints (GroundedSAM)
pixi run download-data
pixi run download-checkpoints
```
*Note: Custom CUDA extensions (`simple-knn`, `diff-plane-rasterization`, `RoPE2D`) are compiled via `pixi run install-submodules` (automatically triggered by `download-data`).*

### 3. Run Pipeline (Recommended)
The easiest way to run the entire pipeline is using the unified `pipeline` task. This handles everything from geometric priors to rendering, including automatic symlinking for Mutagen-style datasets.

```bash
# General Usage
pixi run pipeline -s <path/to/dataset> -t "<text prompts>"

# Example (Mutagen)
pixi run pipeline -s /workspace/mutagen/ultimate_frames -m output/mutagen -t "wall. floor. door. screen. window. ceiling. table"
```

**Optional Arguments:**
- `--image_size <int>`: Inference size for DUSt3R (default: 512).
- `--group_size <int>`: Number of images per group for DUSt3R (default: 10).
- `--skip_geomprior`, `--skip_lp3`, `--skip_train`, `--skip_render`: Skip specific stages.

### 4. Run Individual Tasks (Advanced)
You can run individual stages manually by passing arguments after `--`.

#### Setup for Mutagen
If using the Mutagen dataset, run this helper first to link the sparse model:
```bash
pixi run setup-mutagen
```

#### Step 1: Geometric Priors (DUSt3R)
```bash
pixi run geomprior -- -s <data_path> --group_size 10 --image_size 512
```

#### Step 2: Language-Prompted Planar Priors (LP3)
```bash
pixi run lp3 -- -s <data_path> -t "wall. floor. window."
```

#### Step 3: Training
```bash
pixi run train -- -s <data_path> -m <output_path>
```

#### Step 4: Rendering
```bash
pixi run render -- -m <output_path>
```

## Directory Structure
- `data/`: Contains the dataset and generated priors.
- `output/`: Contains the trained model and renders.
- `pixi_scripts/`: Helper scripts for data download and setup.
- `pixi.toml`: Configuration for the environment and tasks.

## Acknowledgements
This code is built on top of [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [DUSt3R](https://github.com/naver/dust3r), and [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything).
- Please download checkpoints of GroundedSAM from [link1](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
) and [link2](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth), and put them into the `ckpt` folder.
## Dataset Preprocess
We evaluate our method on multi-view images from three indoor datasets:

- [Replica](https://github.com/facebookresearch/Replica-Dataset/): We use eight scenes (office0–office4 and room0–room2), sampling 100 views from each scene.

- [ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/): We select four DSLR-captured sequences: 8b5caf3398, b20a261fdf, 66c98f4a9b, and 88cf747085.

- [MuSHRoom](https://xuqianren.github.io/publications/MuSHRoom/): Our experiments include five iPhone-captured short sequences: coffee_room, classroom, honka, kokko, and vr_room.

We provide all the above above data preprocessed by COLMAP, which can be downloaded from [Google Drive](https://drive.google.com/file/d/1HsgHZt23ECoug8WTRQVHviGu9h5HgSA9/view?usp=sharing) or the `PlanarGS_dataset` folder of our [Hugging Face Datasets](https://huggingface.co/datasets/Isabellaxr/PlanarGS_dataset/tree/main). Starting from these data, you can skip the alignment calculation to GT mesh and conveniently evaluate the reconstructed mesh.

**❗Custom Data :** \
If you want to try PlanarGS on other scenes, please use [COLMAP](https://colmap.github.io/) to obtain camera poses and sparse point cloud from multi-view images, and organize the COLMAP results into the **images** and **sparse** directories as shown in our overview of data directory below.
### Generation of Geometric Priors
We use the pre-trained multi-view foundational model [DUSt3R](https://github.com/naver/dust3r) (code is in the `submodule` folder) to generate geometric priors. Please download the checkpoints of DUSt3R from [link3](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth) and put it into the `ckpt` folder.
```shell
# data_path represents the path to a scene folder of a dataset.
python run_geomprior.py -s <data_path> --group_size 40 #--vis
```
- By default, we sample and extract **40** images per group to run DUSt3R. If your GPU has limited memory (e.g., RTX 3090 with 24GB VRAM), setting `--group_size 25` can help reduce memory usage. However, this may slightly reduce the accuracy of DUSt3R and consequently impact the quality of PlanarGS reconstruction.
- DUSt3R can be swapped out for another multi-view foundation model by adding the model to the `submodules` directory and writing the corresponding `./geomprior/run_dust3r.py` code.
### Pipeline for Language-prompted planar priors (LP3)
One of the advantages of using the open-vocabulary foundation model is that, for the scene-specific training of PlanarGS, you can freely design **prompts** tailored to the characteristics of each scene, which may further improve the LP3 pipeline and enhance the reconstruction performance of PlanarGS. 
- The prompts provided with the `-t` option below are suitable for most indoor scenes. 
- You may also add or remove prompts according to **the planar objects** present in the scene, especially for planes that appear curved in the reconstructed meshes.
```shell
python run_lp3.py -s <data_path> -t "wall. floor. door. screen. window. ceiling. table" #--vis
```
- GroundedSAM can be swapped out for another vision-language foundation model by adding the model to the `submodules` directory and writing the corresponding `./lp3/run_groundedsam.py` code.
### Overview of Data Directory
The data directory after preprocession should contain the following components to be complete for training.
```shell
└── <data_path>
    ├── images
    ├── sparse
    │   ├── cameras.bin
    │   ├── images.bin
    │   └── points3D.bin
    ├── geomprior
    │   ├── aligned_depth
    │   ├── resized_confs
    │   ├── prior_normal
    │   └── depth_weights.json
    └── planarprior
        └── mask
```

## Training and Evaluation
Run `train.py` for 30,000 iterations to obtain the Gaussian reconstruction result `point_cloud.ply`. Then run `render.py` to render color images, depth maps, and normal maps from the reconstructed Gaussians, and generate a mesh `tsdf_fusion_post.ply` using the TSDF method. (The meshes can be viewed with [MeshLab](https://www.meshlab.net/)).

- For mesh generation, you can adjust the parameters `--voxel_size` and `--max_depth` according to the scene.
- The `--eval` option splits the scene into training and test sets for novel view synthesis evaluation.

```shell
python train.py -s <data_path> -m <output_path>  #--eval
python render.py -m <output_path> --voxel_size 0.02 --max_depth 100.0  #--eval
```
If you enable `--eval` during training and rendering, you can run `metrics.py` to evaluate the quality of novel view synthesis.
```shell
python metrics.py -m <output_path>
```
### Evaluation of Reconstructed Mesh
We provide a comprehensive evaluation pipeline including **alignment** and **metric calculation**. The evaluation consists of two steps:

#### 1. Alignment Preprocessing

**Quick Start (Pre-computed Alignment):**
For the datasets used in our paper (Replica, ScanNet++, and MuSHRoom), if you start from our COLMAP-processed data, we provide pre-calculated alignment files `align_params.npz` to the GT mesh `mesh.ply`.
1. Download them from the `align_info` folder of our [Hugging Face Dataset](https://huggingface.co/datasets/Isabellaxr/PlanarGS_dataset/tree/main).
2. Place the `align_params.npz` and `mesh.ply` file into the <data_path> of each scene.
3. **Skip this step** and proceed directly to **Step 2: Metric Calculation**.

**For Custom Data:**
<!-- If you are evaluating on a new scene or a custom dataset, you must run the alignment script to generate `align_params.npz`. This script calculates the coarse alignment (scale & transform) and performs fine-grained ICP registration. -->
If you are evaluating on a new scene or want to run the alignment from scratch, you should have the ground truth
data (including GT mesh, depth maps, and poses) to calculate the scale and coordinate transformation.
- For Replica, ScanNet++, and MuSHRoom, we provide the required GT data structure in the `align_gt` folder of our [Hugging Face Dataset](https://huggingface.co/datasets/Isabellaxr/PlanarGS_dataset/tree/main). Please download and extract it (e.g., to `align_gt_path`).
- For your own custom dataset, please organize your GT data to match the structure expected by the script (refer to `eval_preprocess.py` for details on required depth/pose files).
- Generate the `align_params.npz` by specifying the `align_gt_path`:
  

```shell
# Available dataset_types: [scannetpp, replica, mushroom]
python eval_preprocess.py -s <data_path> -m <output_path> --dataset_type <dataset_type> --gt_data_path <align_gt_path>
```

#### 2. Metric Calculation

Once aligned, run the evaluation script to compute reconstruction metrics.

**For PlanarGS:**
```shell
python eval_recon.py -s <data_path> -m <output_path>
```

**For Other Methods (e.g., 2DGS, PGSR, DN-Splatter):** 
Our evaluation script supports comparing other methods by specifying the method name and mesh path. Note: For `dn_splatter`, we automatically apply necessary coordinate system fixes.
```
python eval_recon.py -s <data_path> -m <output_path> \
    --method 2dgs \
    --rec_mesh_path /path/to/other/mesh.ply
```
## Acknowledgements
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [PGSR](https://github.com/zju3dv/PGSR), and evaluation scripts are based on [NICE-SLAM](https://github.com/cvg/nice-slam). For the usage of the foundation models, we make modifications on the demo code of [DUSt3R](https://github.com/naver/dust3r) and [GroundedSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything). We thank the authors for their great work and repos. 

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{jin2025planargs,
  title     = {PlanarGS: High-Fidelity Indoor 3D Gaussian Splatting Guided by Vision-Language Planar Priors},
  author    = {Xirui Jin and Renbiao Jin and Boying Li and Danping Zou and Wenxian Yu},
  year      = {2025},
  booktitle = {Proceedings of the 39th International Conference on Neural Information Processing Systems}
}
```