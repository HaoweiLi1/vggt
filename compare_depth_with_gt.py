#!/usr/bin/env python3
"""
深度图比较工具：比较两个VGGT模型输出与真实值（Ground Truth）
基于 comparedepth.py 的核心功能，专门用于当前的文件结构
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.gridspec as gridspec
import argparse
import sys
import json

# 配置matplotlib以支持中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_depth_map(path):
    """从PNG文件加载深度图"""
    try:
        img = Image.open(path)
        depth_array = np.array(img).astype(np.float32)
        
        # 如果需要转换为灰度图
        if len(depth_array.shape) == 3:
            depth_array = np.mean(depth_array[:, :, :3], axis=2)
        
        return depth_array
    except Exception as e:
        print(f"加载深度图失败 {path}: {e}")
        return None

def load_rgb_image(path):
    """加载RGB图像"""
    try:
        return np.array(Image.open(path))
    except Exception as e:
        print(f"加载RGB图像失败 {path}: {e}")
        return None

def normalize_depth_maps(depth1, depth2):
    """将两个深度图标准化到相同的尺度"""
    # 如果尺寸不匹配则调整大小
    if depth1.shape != depth2.shape:
        h, w = depth2.shape
        depth1_pil = Image.fromarray(depth1)
        depth1 = np.array(depth1_pil.resize((w, h), Image.BILINEAR))
    
    # 标准化到0-1范围
    depth1_norm = (depth1 - depth1.min()) / (depth1.max() - depth1.min() + 1e-8)
    depth2_norm = (depth2 - depth2.min()) / (depth2.max() - depth2.min() + 1e-8)
    
    return depth1_norm, depth2_norm

def compute_metrics(depth_map, reference_map):
    """计算深度图与参考图之间的差异指标"""
    # 标准化两个图
    depth_norm, ref_norm = normalize_depth_maps(depth_map, reference_map)
    
    # 计算差异
    diff_map = np.abs(depth_norm - ref_norm)
    
    # 计算指标
    mae = np.mean(diff_map)
    rmse = np.sqrt(np.mean(diff_map ** 2))
    correlation = np.corrcoef(depth_norm.flatten(), ref_norm.flatten())[0, 1]
    
    return {
        'diff_map': diff_map,
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation
    }

def create_gt_comparison_visualization(rgb_image, gt_depth, model_depths, model_names, metrics, output_path):
    """创建与真实值的比较可视化"""
    # 创建图形：4列（RGB + GT + 2个模型），2行
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.15, hspace=0.25)
    
    # 绘制RGB图像（跨越两行）
    ax_rgb = fig.add_subplot(gs[:, 0])
    if rgb_image is not None:
        ax_rgb.imshow(rgb_image)
        ax_rgb.set_title('RGB Image', fontsize=14, fontweight='bold')
    else:
        ax_rgb.text(0.5, 0.5, 'RGB Image\nNot Found', ha='center', va='center', fontsize=12)
        ax_rgb.set_title('RGB Image', fontsize=14, fontweight='bold')
    ax_rgb.axis('off')
    
    # 绘制真实值深度图（跨越两行）
    ax_gt = fig.add_subplot(gs[:, 1])
    im_gt = ax_gt.imshow(gt_depth, cmap='viridis')
    ax_gt.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax_gt.axis('off')
    plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
    
    # 绘制每个模型的深度图和差异图
    for i, (depth_map, name, metric) in enumerate(zip(model_depths, model_names, metrics)):
        col = i + 2
        
        # 上排：模型深度图
        ax_depth = fig.add_subplot(gs[0, col])
        im_depth = ax_depth.imshow(depth_map, cmap='viridis')
        ax_depth.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax_depth.axis('off')
        
        # 下排：与真实值的差异图和指标
        ax_diff = fig.add_subplot(gs[1, col])
        im_diff = ax_diff.imshow(metric['diff_map'], cmap='hot', vmin=0, vmax=0.5)
        
        # 在左上角添加指标信息
        metrics_text = f"MAE: {metric['mae']:.3f}\nRMSE: {metric['rmse']:.3f}\nCorr: {metric['correlation']:.3f}"
        ax_diff.text(0.05, 0.95, metrics_text, transform=ax_diff.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top', fontsize=9)
        
        ax_diff.set_title('Difference from GT', fontsize=10)
        ax_diff.axis('off')
        
        # 为最后一列添加颜色条
        if col == 3:
            plt.colorbar(im_depth, ax=ax_depth, fraction=0.046, pad=0.04)
            cbar = plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)
            cbar.set_label('Absolute Difference', fontsize=8)
    
    plt.suptitle('VGGT Depth Map Comparison with Ground Truth', fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"比较可视化已保存到: {output_path}")

def load_gt_camera_params(gt_path):
    """加载GT相机参数文件（4x4矩阵格式）"""
    try:
        with open(gt_path, 'r') as f:
            data = f.read().strip()
        
        # 解析为浮点数数组
        values = np.array([float(x) for x in data.split()])
        
        if len(values) != 16:
            raise ValueError(f"Expected 16 values for 4x4 matrix, got {len(values)}")
        
        # 重塑为4x4矩阵
        matrix_4x4 = values.reshape(4, 4)
        return matrix_4x4
    except Exception as e:
        print(f"加载GT相机参数失败 {gt_path}: {e}")
        return None

def convert_gt_to_standard_format(gt_matrix_4x4):
    """将4x4 GT矩阵转换为标准的3x4外参格式"""
    if gt_matrix_4x4 is None:
        return None
    
    # 提取3x4外参部分（旋转和平移）
    extrinsic_3x4 = gt_matrix_4x4[:3, :]
    
    return {
        "extrinsic": extrinsic_3x4.tolist(),
        "intrinsic": None  # GT文件中没有内参信息
    }

def load_camera_params(cam_params_path):
    """加载相机参数JSON文件"""
    try:
        with open(cam_params_path, 'r') as f:
            params = json.load(f)
        
        # 获取第一帧的参数
        frame_key = list(params.keys())[0]
        return params[frame_key]
    except Exception as e:
        print(f"加载相机参数失败 {cam_params_path}: {e}")
        return None

def compute_camera_metrics(params1, params2):
    """计算两组相机参数之间的差异指标"""
    if params1 is None or params2 is None:
        return None
    
    # 提取外参矩阵
    ext1 = np.array(params1["extrinsic"])  # 3x4
    ext2 = np.array(params2["extrinsic"])  # 3x4
    
    # 提取旋转矩阵和平移向量
    R1, t1 = ext1[:3, :3], ext1[:3, 3]
    R2, t2 = ext2[:3, :3], ext2[:3, 3]
    
    # 计算旋转角度差异（使用Rodrigues公式）
    R_diff = R1.T @ R2  # 相对旋转矩阵
    trace_R = np.trace(R_diff)
    # 确保trace在有效范围内
    trace_R = np.clip(trace_R, -1, 3)
    rotation_angle_rad = np.arccos((trace_R - 1) / 2)
    rotation_angle_deg = np.degrees(rotation_angle_rad)
    
    # 计算平移距离差异
    translation_diff = np.linalg.norm(t1 - t2)
    
    # 计算内参差异（如果都有内参）
    focal_length_diff = None
    if params1.get("intrinsic") is not None and params2.get("intrinsic") is not None:
        intr1 = np.array(params1["intrinsic"])
        intr2 = np.array(params2["intrinsic"])
        fx_diff = abs(intr1[0, 0] - intr2[0, 0])
        fy_diff = abs(intr1[1, 1] - intr2[1, 1])
        focal_length_diff = (fx_diff + fy_diff) / 2
    
    return {
        'rotation_angle_deg': rotation_angle_deg,
        'translation_distance': translation_diff,
        'focal_length_diff': focal_length_diff
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="比较VGGT模型输出与真实值")
    parser.add_argument("--gt_path", type=str, default="depth000000.png",
                        help="真实值深度图路径")
    parser.add_argument("--standard_path", type=str, default="vggt_output/frame000000_depth.png",
                        help="VGGT标准模型输出路径")
    parser.add_argument("--commercial_path", type=str, default="vggt_output_commercial/frame000000_depth.png",
                        help="VGGT商业模型输出路径")
    parser.add_argument("--rgb_path", type=str, default="frame000000.jpg",
                        help="RGB参考图像路径")
    parser.add_argument("--output", type=str, default="depth_comparison.png",
                        help="输出比较图像路径")
    parser.add_argument("--gt_cam_path", type=str, default="gt_cam_params.json",
                        help="真实值相机参数路径（4x4矩阵格式）")
    parser.add_argument("--compare_camera", action="store_true",
                        help="是否比较相机参数")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    required_files = [
        ("真实值深度图", args.gt_path),
        ("VGGT标准模型输出", args.standard_path),
        ("VGGT商业模型输出", args.commercial_path)
    ]
    
    missing_files = []
    for name, path in required_files:
        if not Path(path).exists():
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print("错误：以下必需文件未找到：")
        for missing in missing_files:
            print(f"  - {missing}")
        sys.exit(1)
    
    # 可选的RGB图像
    rgb_image = None
    if Path(args.rgb_path).exists():
        rgb_image = load_rgb_image(args.rgb_path)
        print(f"已加载RGB图像: {args.rgb_path}")
    else:
        print(f"警告：RGB图像未找到: {args.rgb_path}")
    
    # 加载深度图
    print("正在加载深度图...")
    gt_depth = load_depth_map(args.gt_path)
    standard_depth = load_depth_map(args.standard_path)
    commercial_depth = load_depth_map(args.commercial_path)
    
    if any(depth is None for depth in [gt_depth, standard_depth, commercial_depth]):
        print("错误：深度图加载失败")
        sys.exit(1)
    
    print(f"真实值深度图尺寸: {gt_depth.shape}")
    print(f"标准模型深度图尺寸: {standard_depth.shape}")
    print(f"商业模型深度图尺寸: {commercial_depth.shape}")
    
    # 计算指标
    print("\n正在计算比较指标...")
    model_depths = [standard_depth, commercial_depth]
    model_names = ['VGGT Standard', 'VGGT Commercial']
    metrics = []
    
    for name, depth in zip(model_names, model_depths):
        metric = compute_metrics(depth, gt_depth)
        metrics.append(metric)
        print(f"{name}:")
        print(f"  MAE: {metric['mae']:.4f}")
        print(f"  RMSE: {metric['rmse']:.4f}")
        print(f"  相关性: {metric['correlation']:.4f}")
    
    # 创建比较可视化
    print(f"\n正在创建比较可视化...")
    create_gt_comparison_visualization(
        rgb_image, gt_depth, model_depths, model_names, metrics, args.output
    )
    
    # 相机参数比较（可选）
    if args.compare_camera and Path(args.gt_cam_path).exists():
        print(f"\n正在比较相机参数...")
        
        # 加载GT相机参数
        gt_matrix = load_gt_camera_params(args.gt_cam_path)
        if gt_matrix is not None:
            gt_cam_params = convert_gt_to_standard_format(gt_matrix)
            
            # 加载模型输出的相机参数
            standard_cam_path = Path(args.standard_path).parent / "cam_params.json"
            commercial_cam_path = Path(args.commercial_path).parent / "cam_params.json"
            
            standard_cam_params = load_camera_params(standard_cam_path)
            commercial_cam_params = load_camera_params(commercial_cam_path)
            
            # 计算相机参数比较指标
            if standard_cam_params and commercial_cam_params:
                standard_metrics = compute_camera_metrics(standard_cam_params, gt_cam_params)
                commercial_metrics = compute_camera_metrics(commercial_cam_params, gt_cam_params)
                
                print("\n相机参数比较结果:")
                print("=" * 50)
                
                if standard_metrics:
                    print(f"VGGT Standard vs Ground Truth:")
                    print(f"  旋转角度差异: {standard_metrics['rotation_angle_deg']:.3f}°")
                    print(f"  平移距离差异: {standard_metrics['translation_distance']:.6f}")
                    if standard_metrics['focal_length_diff'] is not None:
                        print(f"  焦距差异: {standard_metrics['focal_length_diff']:.3f}")
                
                if commercial_metrics:
                    print(f"VGGT Commercial vs Ground Truth:")
                    print(f"  旋转角度差异: {commercial_metrics['rotation_angle_deg']:.3f}°")
                    print(f"  平移距离差异: {commercial_metrics['translation_distance']:.6f}")
                    if commercial_metrics['focal_length_diff'] is not None:
                        print(f"  焦距差异: {commercial_metrics['focal_length_diff']:.3f}")
                        
                print("=" * 50)
            else:
                print("警告：无法加载模型相机参数进行比较")
        else:
            print("警告：无法加载GT相机参数")
    elif args.compare_camera:
        print(f"警告：GT相机参数文件未找到: {args.gt_cam_path}")

    print(f"\n处理完成！")
    print(f"输出文件: {args.output}")
    if args.compare_camera:
        print(f"相机参数比较: {'已完成' if Path(args.gt_cam_path).exists() else '已跳过（文件未找到）'}")

if __name__ == "__main__":
    main()
