## ___***ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis***___
<div align="center">
<img src='assets/logo.png' style="height:100px"></img>

 <a href='https://arxiv.org/abs/2409.02048'><img src='https://img.shields.io/badge/arXiv-2409.02048-b31b1b.svg'></a> &nbsp;
 <a href='https://drexubery.github.io/ViewCrafter/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
 <a href='https://www.youtube.com/watch?v=WGIEmu9eXmU'><img src='https://img.shields.io/badge/Youtube-Video-b31b1b.svg'></a>&nbsp;
 <a href='https://huggingface.co/spaces/Doubiiu/ViewCrafter'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> &nbsp;

_**[Wangbo Yu*](https://scholar.google.com/citations?user=UOE8-qsAAAAJ&hl=zh-CN), [Jinbo Xing*](https://doubiiu.github.io/), [Li Yuan&dagger;](), [Wenbo Hu&dagger;](https://wbhu.github.io/), [Xiaoyu Li](https://xiaoyu258.github.io/), [Zhipeng Huang](), <br> [Xiangjun Gao](https://scholar.google.com/citations?user=qgdesEcAAAAJ&hl=en/), [Tien-Tsin Wong](https://www.cse.cuhk.edu.hk/~ttwong/myself.html), [Ying Shan](https://scholar.google.com/citations?hl=en&user=4oXBp9UAAAAJ&view_op=list_works&sortby=pubdate), [Yonghong Tian&dagger;]()**_
<br>

<strong>Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 2025</strong>
<br>
</div>

🤗 If you find ViewCrafter useful, **please help ⭐ this repo**, which is important to Open-Source projects. Thanks!

## 🔆 Introduction

- __[2024-11-6]__: Add a simple evaluation script for single-view novel view synthesis.
- __[2024-10-15]__: 🔥🔥 Release the code for sparse-view novel view synthesis.
- __[2024-09-01]__: Launch the project page and update the arXiv preprint.
- __[2024-09-01]__: Release pretrained models and the code for single-view novel view synthesis.

ViewCrafter can generate high-fidelity novel views from <strong>a single or sparse reference image</strong>, while also supporting highly precise pose control. Below shows some examples:

### Zero-shot novel view synthesis (single view)
<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Reference image</td>
        <td>Camera trajecotry</td>
        <td>Generated novel view video</td>
    </tr>

   <tr>
  <td>
    <img src=assets/train.png width="250">
  </td>
  <td>
    <img src=assets/ctrain.gif width="150">
  </td>
  <td>
    <img src=assets/train.gif width="250">
  </td>
  </tr>
  <tr>
  <td>
    <img src=assets/wst.png width="250">
  </td>
  <td>
    <img src=assets/cwst.gif width="150">
  </td>
  <td>
    <img src=assets/wst.gif width="250">
  </td>
  </tr> 
  <tr>
  <td>
    <img src=assets/flower.png width="250">
  </td>
  <td>
    <img src=assets/cflower.gif width="150">
  </td>
  <td>
    <img src=assets/flower.gif width="250">
  </td>
  </tr>
</table>

### Zero-shot novel view synthesis (two views)
<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Reference image 1</td>
        <td>Reference image 2</td>
        <td>Generated novel view video</td>
    </tr>

   <tr>
  <td>
    <img src=assets/car2_1.png width="250">
  </td>
  <td>
    <img src=assets/car2_2.png width="250">
  </td>
  <td>
    <img src=assets/car2.gif width="250">
  </td>
  </tr>
  <tr>
  <td>
    <img src=assets/barn_0.png width="250">
  </td>
  <td>
    <img src=assets/barn_2.png width="250">
  </td>
  <td>
    <img src=assets/barn.gif width="250">
  </td>
  </tr> 
  <tr>
  <td>
    <img src=assets/house_1.png width="250">
  </td>
  <td>
    <img src=assets/house_2.png width="250">
  </td>
  <td>
    <img src=assets/house.gif width="250">
  </td>
  </tr>
</table>


## 🧰 Models

|Model|Resolution|Frames|GPU Mem. & Inference Time (tested on a 40G A100, ddim 50 steps)|Checkpoint|Description|
|:---------|:---------|:--------|:--------|:--------|:--------|
|ViewCrafter_25|576x1024|25| 23.5GB & 120s (`perframe_ae=True`)|[Hugging Face](https://huggingface.co/Drexubery/ViewCrafter_25/blob/main/model.ckpt)|Used for single view NVS, can also adapt to sparse view NVS|
|ViewCrafter_25_sparse|576x1024|25| 23.5GB & 120s (`perframe_ae=True`)|[Hugging Face](https://huggingface.co/Drexubery/ViewCrafter_25_sparse/blob/main/model_sparse.ckpt)|Used for sparse view NVS|
|ViewCrafter_16|576x1024|16| 18.3GB & 75s (`perframe_ae=True`)|[Hugging Face](https://huggingface.co/Drexubery/ViewCrafter_16/blob/main/model.ckpt)|16 frames model, used for ablation|
|ViewCrafter_25_512|320x512|25| 13.8GB & 50s (`perframe_ae=True`)|[Hugging Face](https://huggingface.co/Drexubery/ViewCrafter_25_512/blob/main/model.ckpt)|512 resolution model, used for ablation|

## ⚙️ Setup

### 1. Clone Our ViewCrafter
```bash
git clone https://github.com/hg-tc/viewcrafter_new.git
cd ViewCrafter
```
### 2. Installation

```bash
# Create conda environment
conda create -n viewcrafter python=3.9.16
conda activate viewcrafter
pip install -r requirements.txt

# Install PyTorch3D
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py39_cu117_pyt1131.tar.bz2

# Download pretrained DUSt3R model
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/

```
Download pretrained ViewCrafter_25 and put the model.ckpt in checkpoints/model.ckpt.
https://huggingface.co/Drexubery/ViewCrafter_25/blob/main/model.ckpt

> [!NOTE]
> If you use a high PyTorch version (like torch 2.4), it may cause CUDA OOM error. Please refer to [these issues](https://github.com/Drexubery/ViewCrafter/issues/23#issuecomment-2396131121) for solutions.

## 💫 Inference 
### 从图片构建视频

```bash
  sh run_all.sh
```
### 根据位姿进行特定位置的视图综合

```bash
  sh run_with_pose.sh
```

脚本介绍
```bash
#!/usr/bin/env bash
set -e

python /home/nics/Workspace/ViewCrafter/image_process/unify_camera_intrinsics_with_crop.py \
# input_dir替换为输入文件夹，包含三张图片和相机内参（见 d435i.yaml）
  --input_dir "/home/nics/Workspace/ViewCrafter/input" \ 
# output_dir替换为相对路径，输出内参归一化图片 与--image_dir一致即可
  --output_dir "/home/nics/Workspace/ViewCrafter/test/three_unified"

python /home/nics/Workspace/ViewCrafter/inference_cam0.py \
  --image_dir test/three_unified \
  --out_dir ./output \
  --mode external_pose \
# 视图综合目标参数矩阵文件
  --ext_pose_path /home/nics/Workspace/ViewCrafter/image_process/ext_pose.json \
# 相机外参，同一相机不需要修改，为相对位姿image_process文件夹有处理脚本，新数据可以再处理
  --real_poses_yaml /home/nics/Workspace/ViewCrafter/image_process/cam_extrinsics.yaml \
  --device cuda:0 \
  --config configs/inference_pvd_1024.yaml \
  --ckpt_path ./checkpoints/model_sparse.ckpt \
  --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --height 576 --width 1024 --video_length 15 \
  --bg_trd 0.2 \
  --seed 123 \
  --ddim_steps 50
```