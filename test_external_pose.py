#!/usr/bin/env python3
"""
测试外部位姿渲染的示例脚本
"""

import json
import torch
import numpy as np

def create_test_pose():
    """创建一个测试位姿（cam0和cam1的中间位置）"""
    # 从真实位姿文件中读取cam0和cam1的位姿
    import yaml
    with open('/home/nics/Workspace/ViewCrafter/image_process/cam_extrinsics.yaml', 'r') as f:
        data = yaml.safe_load(f)
    
    cam0_pose = torch.tensor(data['cam0']['T_cam_to_world_4x4'], dtype=torch.float32)
    cam1_pose = torch.tensor(data['cam1']['T_cam_to_world_4x4'], dtype=torch.float32)
    
    # 简单线性插值
    alpha = 0.5
    interpolated_pose = (1 - alpha) * cam0_pose + alpha * cam1_pose
    
    # 确保旋转矩阵的正交性（简单处理）
    R = interpolated_pose[:3, :3]
    U, S, V = torch.svd(R)
    R_orthogonal = U @ V.T
    interpolated_pose[:3, :3] = R_orthogonal
    
    return interpolated_pose.numpy().tolist()

def main():
    # 创建测试位姿
    test_pose = create_test_pose()
    
    # 保存为JSON文件
    pose_file = '/home/nics/Workspace/ViewCrafter/test_external_pose.json'
    with open(pose_file, 'w') as f:
        json.dump(test_pose, f, indent=2)
    
    print(f"测试位姿已保存到: {pose_file}")
    print("位姿矩阵:")
    for row in test_pose:
        print(f"  {row}")
    
    # 运行外部位姿渲染
    import subprocess
    cmd = [
        'python', '/home/nics/Workspace/ViewCrafter/inference_cam0.py',
        '--image_dir', 'test/three_unified',
        '--out_dir', './output',
        '--mode', 'external_pose',
        '--ext_pose_path', pose_file,
        '--real_poses_yaml', '/home/nics/Workspace/ViewCrafter/image_process/cam_extrinsics.yaml',
        '--device', 'cuda:0',
        '--config', 'configs/inference_pvd_1024.yaml',
        '--ckpt_path', './checkpoints/model_sparse.ckpt',
        '--model_path', './checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth',
        '--height', '576', '--width', '1024', '--video_length', '25'
    ]
    
    print(f"\n运行命令: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
