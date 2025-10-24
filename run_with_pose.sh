#!/usr/bin/env bash
set -e

python /home/nics/Workspace/ViewCrafter/image_process/unify_camera_intrinsics_with_crop.py \
  --input_dir "/home/nics/Workspace/ViewCrafter/input" \
  --output_dir "/home/nics/Workspace/ViewCrafter/test/three_unified"

# python /home/nics/Workspace/ViewCrafter/inference_cam0.py \
# --image_dir test/three_unified \
# --out_dir ./output \
# --mode 'external_pose' \
# --bg_trd 0.2 \
# --seed 123 \
# --interp_alpha 0.5 \
# --ckpt_path ./checkpoints/model_sparse.ckpt \
# --config configs/inference_pvd_1024.yaml \
# --ddim_steps 50 \
# --video_length 15 \
# --device 'cuda:0' \
# --height 576 --width 1024 \
# --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

python /home/nics/Workspace/ViewCrafter/inference_cam0.py \
  --image_dir test/three_unified \
  --out_dir ./output \
  --mode external_pose \
  --ext_pose_path /home/nics/Workspace/ViewCrafter/image_process/ext_pose.json \
  --real_poses_yaml /home/nics/Workspace/ViewCrafter/image_process/cam_extrinsics.yaml \
  --device cuda:0 \
  --config configs/inference_pvd_1024.yaml \
  --ckpt_path ./checkpoints/model_sparse.ckpt \
  --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --height 576 --width 1024 --video_length 15 \
  --bg_trd 0.2 \
  --seed 123 \
  --ddim_steps 50