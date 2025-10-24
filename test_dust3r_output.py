#!/usr/bin/env python3
"""
测试DUSt3R输出功能的脚本
"""

import os
import sys
from viewcrafter import ViewCrafter
from configs.infer_config import get_parser
from datetime import datetime
import json
import numpy as np

def test_dust3r_output():
    """测试DUSt3R输出功能"""
    
    # 设置测试参数
    parser = get_parser()
    
    # 使用测试图像
    test_image_dir = "test/three"  # 使用现有的测试图像
    
    # 设置基本参数
    test_args = [
        '--image_dir', test_image_dir,
        '--mode', 'sparse_view_interp',
        '--out_dir', 'output',
        '--exp_name', f'test_dust3r_output_{datetime.now().strftime("%Y%m%d_%H%M")}',
        '--batch_size', '1',
        '--niter', '3',
        '--schedule', 'cosine',
        '--lr', '0.01',
        '--dpt_trd', '1.0',
        '--min_conf_thr', '0.1',
        '--bg_trd', '0.1',
        '--center_scale', '1.0',
        '--elevation', '0.0',
        '--video_length', '25',
        '--height', '576',
        '--width', '1024',
        '--bs', '1',
        '--ddim_steps', '50',
        '--ddim_eta', '0.0',
        '--unconditional_guidance_scale', '7.5',
        '--cfg_img', '1.0',
        '--frame_stride', '1',
        '--timestep_spacing', 'uniform_trailing',
        '--guidance_rescale', '0.0',
        '--seed', '42',
        '--perframe_ae', 'True',
        '--model_path', 'checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth',
        '--ckpt_path', 'checkpoints/model_sparse.ckpt',
        '--config', 'configs/inference_pvd_512.yaml'
    ]
    
    # 临时替换sys.argv
    original_argv = sys.argv
    sys.argv = ['test_dust3r_output.py'] + test_args
    opts = parser.parse_args()
    sys.argv = original_argv
    
    # 创建输出目录
    opts.save_dir = os.path.join(opts.out_dir, opts.exp_name)
    os.makedirs(opts.save_dir, exist_ok=True)
    
    print(f"开始测试DUSt3R输出功能...")
    print(f"输入图像目录: {opts.image_dir}")
    print(f"输出目录: {opts.save_dir}")
    
    try:
        # 创建ViewCrafter实例
        pvd = ViewCrafter(opts)
        
        print("DUSt3R处理完成，检查输出文件...")
        
        # 检查输出文件
        dust3r_output_dir = os.path.join(opts.save_dir, "dust3r_output")
        
        if os.path.exists(dust3r_output_dir):
            print(f"✓ DUSt3R输出目录已创建: {dust3r_output_dir}")
            
            # 检查各个文件
            files_to_check = [
                "pointcloud.ply",
                "camera_parameters.json", 
                "summary.json"
            ]
            
            dirs_to_check = [
                "depth_maps",
                "input_images"
            ]
            
            for file in files_to_check:
                file_path = os.path.join(dust3r_output_dir, file)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"✓ {file} 已保存 ({file_size} bytes)")
                else:
                    print(f"✗ {file} 未找到")
            
            for dir_name in dirs_to_check:
                dir_path = os.path.join(dust3r_output_dir, dir_name)
                if os.path.exists(dir_path):
                    file_count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
                    print(f"✓ {dir_name} 目录已创建 ({file_count} 个文件)")
                else:
                    print(f"✗ {dir_name} 目录未找到")
            
            print(f"\n测试完成！所有DUSt3R输出文件已保存到: {dust3r_output_dir}")

            # 自动尝试生成米制深度：优先用 METRIC_CAM_PARAMS；否则在常见路径查找；再否则用 BASELINE_METERS
            def _find_metric_cam_path():
                # 1) 环境变量
                p = os.environ.get('METRIC_CAM_PARAMS', '').strip()
                if p and os.path.exists(p):
                    return p
                # 2) 常见文件名在图像目录/输出目录/项目根目录
                candidates = [
                    os.path.join(opts.image_dir, 'real_camera_parameters.json'),
                    os.path.join(dust3r_output_dir, 'real_camera_parameters.json'),
                    os.path.join(os.getcwd(), 'real_camera_parameters.json'),
                ]
                for c in candidates:
                    if os.path.exists(c):
                        return c
                return ''

            metric_cam_path = _find_metric_cam_path()
            if metric_cam_path:
                try:
                    print("\n检测到 METRIC_CAM_PARAMS，开始计算尺度并生成米制深度...")

                    def _load_centers_from_camjson(path):
                        with open(path, 'r') as f:
                            data = json.load(f)
                        frames = data.get('frames', [])
                        centers = []
                        for fr in frames:
                            T = np.array(fr.get('cam2world'), dtype=np.float32)
                            centers.append(T[:3, 3])
                        return np.stack(centers, axis=0) if centers else np.zeros((0, 3), dtype=np.float32)

                    def _umeyama_scale(X, Y):
                        """仅估计尺度 s，使得 Y ≈ s * R * X + t（忽略返回 R,t）。
                        X, Y: (N,3)
                        """
                        assert X.shape == Y.shape and X.ndim == 2 and X.shape[1] == 3
                        n = X.shape[0]
                        if n < 2:
                            return 1.0
                        muX = X.mean(axis=0)
                        muY = Y.mean(axis=0)
                        Xc = X - muX
                        Yc = Y - muY
                        cov = (Yc.T @ Xc) / n
                        U, S, Vt = np.linalg.svd(cov)
                        R = U @ Vt
                        if np.linalg.det(R) < 0:
                            Vt[-1, :] *= -1
                            R = U @ Vt
                        varX = (Xc**2).sum() / n
                        if varX <= 1e-12:
                            return 1.0
                        s = (S.sum()) / varX
                        return float(s)

                    est_cam_json = os.path.join(dust3r_output_dir, "camera_parameters.json")
                    if not os.path.exists(est_cam_json):
                        print("未找到 DUSt3R 相机参数文件，跳过米制深度生成。")
                    else:
                        C_est = _load_centers_from_camjson(est_cam_json)
                        C_real = _load_centers_from_camjson(metric_cam_path)
                        if len(C_est) == 0 or len(C_real) == 0:
                            print("相机中心数量为0，跳过米制深度生成。")
                        else:
                            # 对齐至相同帧数（若数量不同，取最小长度的前缀）
                            n = min(len(C_est), len(C_real))
                            s = _umeyama_scale(C_est[:n], C_real[:n])
                            print(f"估计尺度因子 s = {s:.6f}")

                            # 保存尺度信息
                            with open(os.path.join(dust3r_output_dir, 'metric_scale.json'), 'w') as f:
                                json.dump({"scale": s}, f, indent=2)

                            # 生成米制深度（处理 .npy）
                            depth_dir = os.path.join(dust3r_output_dir, "depth_maps")
                            out_depth_dir = os.path.join(dust3r_output_dir, "depth_maps_metric")
                            os.makedirs(out_depth_dir, exist_ok=True)
                            converted = 0
                            if os.path.exists(depth_dir):
                                for fname in os.listdir(depth_dir):
                                    if fname.lower().endswith('.npy'):
                                        in_path = os.path.join(depth_dir, fname)
                                        out_path = os.path.join(out_depth_dir, fname)
                                        try:
                                            depth = np.load(in_path)
                                            np.save(out_path, depth * s)
                                            converted += 1
                                        except Exception as e:
                                            print(f"转换 {fname} 失败: {e}")
                            print(f"米制深度保存完成，npy 转换数量: {converted}")
                except Exception as e:
                    print(f"米制深度生成失败: {e}")
            else:
                # 回退：若提供 BASELINE_METERS（真实前两帧相机中心距离，单位米），用基线求尺度
                baseline_str = os.environ.get('BASELINE_METERS', '').strip()
                if baseline_str:
                    try:
                        B_real = float(baseline_str)
                        print(f"\n检测到 BASELINE_METERS={B_real}，基于前两帧估计尺度...")
                        est_cam_json = os.path.join(dust3r_output_dir, "camera_parameters.json")
                        if os.path.exists(est_cam_json):
                            with open(est_cam_json, 'r') as f:
                                est = json.load(f)
                            frames = est.get('frames', [])
                            if len(frames) >= 2:
                                T1 = np.array(frames[0]['cam2world'], dtype=np.float32)
                                T2 = np.array(frames[1]['cam2world'], dtype=np.float32)
                                C1 = T1[:3, 3]
                                C2 = T2[:3, 3]
                                B_est = float(np.linalg.norm(C1 - C2))
                                if B_est > 1e-12:
                                    s = B_real / B_est
                                    with open(os.path.join(dust3r_output_dir, 'metric_scale.json'), 'w') as f:
                                        json.dump({"scale": s, "source": "baseline"}, f, indent=2)
                                    depth_dir = os.path.join(dust3r_output_dir, "depth_maps")
                                    out_depth_dir = os.path.join(dust3r_output_dir, "depth_maps_metric")
                                    os.makedirs(out_depth_dir, exist_ok=True)
                                    converted = 0
                                    if os.path.exists(depth_dir):
                                        for fname in os.listdir(depth_dir):
                                            if fname.lower().endswith('.npy'):
                                                in_path = os.path.join(depth_dir, fname)
                                                out_path = os.path.join(out_depth_dir, fname)
                                                try:
                                                    depth = np.load(in_path)
                                                    np.save(out_path, depth * s)
                                                    converted += 1
                                                except Exception as e:
                                                    print(f"转换 {fname} 失败: {e}")
                                    print(f"米制深度保存完成（基线法），npy 转换数量: {converted}")
                                else:
                                    print("估计基线长度过小，跳过米制深度生成。")
                            else:
                                print("相机帧不足2帧，无法用基线估计尺度。")
                        else:
                            print("未找到 DUSt3R 相机参数文件，无法用基线法生成米制深度。")
                    except Exception as e:
                        print(f"基线法米制深度生成失败: {e}")
            
        else:
            print(f"✗ DUSt3R输出目录未创建: {dust3r_output_dir}")
            
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dust3r_output()
