#!/usr/bin/env bash
set -e

# 单帧快速路径：不跑完整 inference，直接在 single_pose_render 内部运行 DUSt3R

# 先统一相机内参（如已有可跳过）
python /home/nics/Workspace/ViewCrafter/image_process/unify_camera_intrinsics_with_crop.py \
  --input_dir "/home/nics/Workspace/ViewCrafter/input" \
  --output_dir "/home/nics/Workspace/ViewCrafter/test/three_unified"

# 直接调用窗口扩散渲染脚本：将目标位姿插入轨迹并用邻域窗口进行扩散
python /home/nics/Workspace/ViewCrafter/single_pose_window_diffusion.py \
  --image_dir "/home/nics/Workspace/ViewCrafter/test/three_unified" \
  --c2w_path "/home/nics/Workspace/ViewCrafter/example_pose.json" \
  --cam_extrinsics "/home/nics/Workspace/ViewCrafter/image_process/cam_extrinsics.yaml" \
  --actual_world cam1 \
  --video_length 25 \
  --device cuda:0

echo "Single pose window diffusion completed. Check the latest folder under /home/nics/Workspace/ViewCrafter/output for single_pose_render.png"


