# 外部位姿渲染功能说明

## 功能概述
新增了坐标转换功能，可以将真实世界坐标系下的外部位姿转换到DUSt3R估计的坐标系下进行渲染。

## 新增功能

### 1. 坐标转换函数
- `_load_real_camera_poses()`: 从YAML文件加载真实相机位姿
- `_estimate_transform_matrix()`: 使用Procrustes分析估计真实位姿到DUSt3R位姿的变换矩阵
- `_transform_external_pose()`: 将外部位姿转换到DUSt3R坐标系

### 2. 新增参数
- `--real_poses_yaml`: 指定真实相机位姿YAML文件路径（默认: `./image_process/cam_extrinsics.yaml`）

## 使用方法

### 方法1: 使用JSON文件指定外部位姿
```bash
python /home/nics/Workspace/ViewCrafter/inference_cam0.py \
  --image_dir test/three_unified \
  --out_dir ./output \
  --mode external_pose \
  --ext_pose_path /path/to/external_pose.json \
  --real_poses_yaml /path/to/real_poses.yaml \
  --device cuda:0 \
  --config configs/inference_pvd_1024.yaml \
  --ckpt_path ./checkpoints/model_sparse.ckpt \
  --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --height 576 --width 1024 --video_length 25
```

### 方法2: 使用插值比例
```bash
python /home/nics/Workspace/ViewCrafter/inference_cam0.py \
  --image_dir test/three_unified \
  --out_dir ./output \
  --mode external_pose \
  --interp_alpha 0.5 \
  --real_poses_yaml /path/to/real_poses.yaml \
  --device cuda:0 \
  --config configs/inference_pvd_1024.yaml \
  --ckpt_path ./checkpoints/model_sparse.ckpt \
  --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --height 576 --width 1024 --video_length 25
```

### 方法3: 使用测试脚本
```bash
python /home/nics/Workspace/ViewCrafter/test_external_pose.py
```

## 文件格式

### JSON位姿文件格式
```json
[
  [1.0, 0.0, 0.0, 0.0],
  [0.0, 1.0, 0.0, 0.0],
  [0.0, 0.0, 1.0, 0.0],
  [0.0, 0.0, 0.0, 1.0]
]
```

### YAML位姿文件格式
```yaml
cam0:
  T_cam_to_world_4x4:
    - [1.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
cam1:
  T_cam_to_world_4x4:
    - [0.866, -0.238, 0.437, 0.114]
    - [0.239, 0.969, 0.055, 0.002]
    - [-0.437, 0.056, 0.897, 0.000]
    - [0.0, 0.0, 0.0, 1.0]
cam2:
  T_cam_to_world_4x4:
    - [0.508, -0.406, 0.758, 0.207]
    - [0.411, 0.888, 0.200, 0.029]
    - [-0.756, 0.210, 0.619, -0.054]
    - [0.0, 0.0, 0.0, 1.0]
```

## 输出文件
- `diffusion_t_target.png`: 目标位姿的渲染结果
- `render.mp4`: 渲染序列视频
- `diffusion.mp4`: 扩散生成视频

## 工作原理
1. 加载真实相机位姿和DUSt3R估计位姿
2. 使用Procrustes分析估计变换矩阵（旋转+平移+缩放）
3. 将输入的外部位姿转换到DUSt3R坐标系
4. 在转换后的位姿进行渲染和扩散生成
