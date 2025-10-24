#!/usr/bin/env bash
set -e

python /home/nics/Workspace/ViewCrafter/image_process/unify_camera_intrinsics_with_crop.py \
  --input_dir "/home/nics/Workspace/ViewCrafter/input" \
  --output_dir "/home/nics/Workspace/ViewCrafter/test/three_unified"

python inference.py \
--image_dir test/three_unified \
--out_dir ./output \
--mode 'sparse_view_interp' \
--bg_trd 0.2 \
--seed 123 \
--ckpt_path ./checkpoints/model_sparse.ckpt \
--config configs/inference_pvd_1024.yaml \
--ddim_steps 50 \
--video_length 25 \
--device 'cuda:0' \
--height 576 --width 1024 \
--model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth