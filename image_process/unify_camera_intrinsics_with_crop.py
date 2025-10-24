#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机内参统一处理脚本（带黑边检测和裁剪）
将三个不同相机的图像统一到相同的内参，并裁剪去除黑边
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
import argparse
import os

def find_camera_params_file(input_dir='.'):
    """自动查找相机参数文件"""
    base = Path(input_dir)
    yaml_files = list(base.glob('*.yaml')) + list(base.glob('*.yml'))
    
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
            
            # 检查是否包含相机参数结构
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict) and 'intrinsics' in value:
                        print(f"找到相机参数文件: {yaml_file}")
                        return str(yaml_file)
        except Exception as e:
            continue
    
    return None

def load_camera_params(yaml_file):
    """加载相机参数"""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    cameras = {}
    for cam_name, params in data.items():
        intrinsics = params['intrinsics']
        cameras[cam_name] = {
            'fx': intrinsics[0],
            'fy': intrinsics[1], 
            'cx': intrinsics[2],
            'cy': intrinsics[3],
            'distortion_coeffs': params['distortion_coeffs']
        }
    
    return cameras

def find_image_files(input_dir='.'):
    """自动查找图像文件"""
    # 常见的图像格式
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(ext))
        image_files.extend(Path(input_dir).glob(ext.upper()))
    
    # 过滤掉已经处理过的文件
    image_files = [f for f in image_files if not str(f).startswith('unified_')]
    
    # 按文件名排序
    image_files.sort()
    
    return [str(f) for f in image_files]

def calculate_unified_intrinsics(cameras):
    """计算统一的内参"""
    # 取所有相机内参的平均值作为统一内参
    fx_values = [cam['fx'] for cam in cameras.values()]
    fy_values = [cam['fy'] for cam in cameras.values()]
    cx_values = [cam['cx'] for cam in cameras.values()]
    cy_values = [cam['cy'] for cam in cameras.values()]
    
    unified = {
        'fx': np.mean(fx_values),
        'fy': np.mean(fy_values),
        'cx': np.mean(cx_values),
        'cy': np.mean(cy_values),
        'distortion_coeffs': [0, 0, 0, 0]  # 假设无畸变
    }
    
    return unified

def create_camera_matrices(intrinsics):
    """创建相机矩阵"""
    K = np.array([
        [intrinsics['fx'], 0, intrinsics['cx']],
        [0, intrinsics['fy'], intrinsics['cy']],
        [0, 0, 1]
    ])
    
    D = np.array(intrinsics['distortion_coeffs'])
    
    return K, D

def detect_valid_region(image, threshold=10):
    """检测图像中的有效区域（非黑边区域）"""
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 创建掩码，非黑色像素为1
    mask = (gray > threshold).astype(np.uint8)
    
    # 找到非零像素的边界
    coords = np.column_stack(np.where(mask > 0))
    
    if len(coords) == 0:
        return None
    
    # 计算边界框
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return (x_min, y_min, x_max, y_max)

def calculate_unified_crop_region(image_files, cameras, unified_intrinsics):
    """计算统一的裁剪区域"""
    print("分析各图像的有效区域...")
    
    valid_regions = []
    
    for img_file, cam_name in zip(image_files, cameras.keys()):
        print(f"  处理 {img_file}...")
        
        # 读取图像
        image = cv2.imread(img_file)
        if image is None:
            print(f"    错误: 无法读取图像 {img_file}")
            continue
            
        h, w = image.shape[:2]
        image_size = (w, h)
        
        # 创建源相机矩阵
        K_src, D_src = create_camera_matrices(cameras[cam_name])
        K_dst, D_dst = create_camera_matrices(unified_intrinsics)
        
        # 重投影图像
        map1, map2 = cv2.initUndistortRectifyMap(
            K_src, D_src, None, K_dst, image_size, cv2.CV_16SC2
        )
        
        undistorted_image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
        
        # 检测有效区域
        valid_region = detect_valid_region(undistorted_image)
        if valid_region is not None:
            valid_regions.append(valid_region)
            print(f"    有效区域: x={valid_region[0]}-{valid_region[2]}, y={valid_region[1]}-{valid_region[3]}")
        else:
            print(f"    警告: 未找到有效区域")
    
    if not valid_regions:
        print("错误: 没有找到任何有效区域")
        return None
    
    # 计算所有有效区域的交集（统一裁剪区域）
    x_min = max(region[0] for region in valid_regions)
    y_min = max(region[1] for region in valid_regions)
    x_max = min(region[2] for region in valid_regions)
    y_max = min(region[3] for region in valid_regions)
    
    # 确保裁剪区域有效
    if x_min >= x_max or y_min >= y_max:
        print("警告: 交集区域无效，使用并集区域")
        x_min = min(region[0] for region in valid_regions)
        y_min = min(region[1] for region in valid_regions)
        x_max = max(region[2] for region in valid_regions)
        y_max = max(region[3] for region in valid_regions)
    
    unified_crop = (x_min, y_min, x_max, y_max)
    print(f"\n统一裁剪区域: x={x_min}-{x_max}, y={y_min}-{y_max}")
    print(f"裁剪后尺寸: {x_max-x_min} x {y_max-y_min}")
    
    return unified_crop

def undistort_and_crop_image(image, K_src, D_src, K_dst, crop_region):
    """对图像进行去畸变、重投影和裁剪"""
    h, w = image.shape[:2]
    image_size = (w, h)
    
    # 计算重投影映射
    map1, map2 = cv2.initUndistortRectifyMap(
        K_src, D_src, None, K_dst, image_size, cv2.CV_16SC2
    )
    
    # 应用重投影
    undistorted = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
    
    # 裁剪
    x_min, y_min, x_max, y_max = crop_region
    cropped = undistorted[y_min:y_max, x_min:x_max]
    
    return cropped

def process_images_with_crop(input_dir='.', output_dir='.'):
    """处理图像的主要函数（带裁剪）"""
    # 自动查找相机参数文件
    yaml_file = find_camera_params_file(input_dir)
    if yaml_file is None:
        print("错误: 未找到相机参数文件")
        print("请确保当前目录下有包含相机内参的YAML文件")
        return None
    
    # 加载相机参数
    cameras = load_camera_params(yaml_file)
    print("原始相机参数:")
    for cam_name, params in cameras.items():
        print(f"{cam_name}: fx={params['fx']:.2f}, fy={params['fy']:.2f}, "
              f"cx={params['cx']:.2f}, cy={params['cy']:.2f}")
    
    # 计算统一内参
    unified_intrinsics = calculate_unified_intrinsics(cameras)
    print(f"\n统一内参: fx={unified_intrinsics['fx']:.2f}, fy={unified_intrinsics['fy']:.2f}, "
          f"cx={unified_intrinsics['cx']:.2f}, cy={unified_intrinsics['cy']:.2f}")
    
    # 自动查找图像文件
    image_files = find_image_files(input_dir)
    if len(image_files) == 0:
        print("错误: 未找到图像文件")
        print("请确保当前目录下有图像文件")
        return None
    
    print(f"\n找到 {len(image_files)} 个图像文件:")
    for img_file in image_files:
        print(f"  {img_file}")
    
    # 获取相机名称列表
    camera_names = list(cameras.keys())
    
    # 确保图像数量与相机数量匹配
    if len(image_files) != len(camera_names):
        print(f"警告: 图像数量({len(image_files)})与相机数量({len(camera_names)})不匹配")
        print("将使用前N个图像文件，其中N=min(图像数量, 相机数量)")
        min_count = min(len(image_files), len(camera_names))
        image_files = image_files[:min_count]
        camera_names = camera_names[:min_count]
    
    # 计算统一裁剪区域
    unified_crop = calculate_unified_crop_region(image_files, cameras, unified_intrinsics)
    if unified_crop is None:
        print("错误: 无法计算统一裁剪区域")
        return None
    
    # 创建统一相机矩阵
    K_unified, D_unified = create_camera_matrices(unified_intrinsics)
    
    # 处理每张图像
    print(f"\n开始处理图像...")
    processed_images = []
    
    for idx, (img_file, cam_name) in enumerate(zip(image_files, camera_names)):
        print(f"处理 {img_file} (对应 {cam_name})...")
        
        # 读取图像
        image = cv2.imread(img_file)
        if image is None:
            print(f"错误: 无法读取图像 {img_file}")
            continue
            
        # 创建源相机矩阵
        K_src, D_src = create_camera_matrices(cameras[cam_name])
        
        # 重投影和裁剪图像
        processed_image = undistort_and_crop_image(
            image, K_src, D_src, K_unified, unified_crop
        )
        
        # 保存结果
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{idx+1:06d}.png")
        cv2.imwrite(output_file, processed_image)
        print(f"保存统一裁剪图像: {output_file}")
        print(f"  尺寸: {processed_image.shape[1]} x {processed_image.shape[0]}")
        
        processed_images.append(processed_image)
    
    # 保存统一的内参到新文件
    unified_params = {
        'unified_camera': {
            'distortion_coeffs': unified_intrinsics['distortion_coeffs'],
            'intrinsics': [
                float(unified_intrinsics['fx']),
                float(unified_intrinsics['fy']),
                float(unified_intrinsics['cx']),
                float(unified_intrinsics['cy'])
            ]
        },
        'crop_region': {
            'x_min': int(unified_crop[0]),
            'y_min': int(unified_crop[1]),
            'x_max': int(unified_crop[2]),
            'y_max': int(unified_crop[3])
        },
        'output_size': {
            'width': int(unified_crop[2] - unified_crop[0]),
            'height': int(unified_crop[3] - unified_crop[1])
        }
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'unified_camera_params_cropped.yaml'), 'w') as f:
        yaml.dump(unified_params, f, default_flow_style=False)
    
    print(f"\n统一内参和裁剪信息已保存到: {os.path.join(output_dir, 'unified_camera_params_cropped.yaml')} ")
    
    return unified_intrinsics, unified_crop, processed_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='相机内参统一处理（带裁剪）')
    parser.add_argument('--input_dir', type=str, default='.', help='输入图片与相机参数所在文件夹（默认当前目录）')
    parser.add_argument('--output_dir', type=str, default='.', help='输出图片与统一参数保存文件夹（默认当前目录）')
    args = parser.parse_args()

    result = process_images_with_crop(input_dir=args.input_dir, output_dir=args.output_dir)
    if result is not None:
        print("\n处理完成!")
        print("所有图像已统一到相同的内参和尺寸，黑边已去除。")
    else:
        print("\n处理失败!")
