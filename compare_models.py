#!/usr/bin/env python3
"""Compare fine-tuned VGGT checkpoints against a baseline."""

import argparse
import contextlib
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from PIL import Image

sys.path.insert(0, 'training')

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# 设置 OpenEXR 支持
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def _extract_state_dict(checkpoint):
    epoch = None
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
            epoch = checkpoint.get("prev_epoch")
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    return state_dict, epoch


def _state_dict_has_vit_fusion(state_dict):
    vit_indicators = (
        "depth_head.compress_vit_early",
        "depth_head.compress_vit_final",
        "depth_head.fusion_conv",
    )
    return any(any(token in key for token in vit_indicators) for key in state_dict.keys())


def load_model_from_pt(model_path, device):
    """Load a VGGT model checkpoint in a device-agnostic way."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict_raw, epoch = _extract_state_dict(checkpoint)

    state_dict = {}
    for key, value in state_dict_raw.items():
        state_dict[key.replace("module.", "", 1) if key.startswith("module.") else key] = value

    use_vit_features = _state_dict_has_vit_fusion(state_dict)
    if use_vit_features:
        print("  Detected ViT fusion weights – enabling use_vit_features")
    else:
        print("  No ViT fusion weights found – using vanilla depth head")

    model = VGGT(
        enable_camera=True,
        enable_depth=True,
        enable_point=False,
        enable_track=False,
        use_vit_features=use_vit_features,
    )

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"  Warning: {len(missing_keys)} missing keys (expected for disabled heads)")
    if unexpected_keys:
        print(f"  Warning: {len(unexpected_keys)} unexpected keys")

    if epoch is not None:
        print(f"  Loaded parameters from epoch {epoch}")

    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model


def predict_depth(model, image_path, device, dtype):
    """使用模型预测深度
    
    Args:
        model: VGGT 模型
        image_path: 图像路径
        device: 设备
        dtype: 数据类型
    
    Returns:
        depth_map: 深度图 numpy 数组 (原始深度值，未归一化)
    """
    images = load_and_preprocess_images([image_path]).to(device)
    
    autocast_enabled = device.startswith("cuda")
    autocast_context = (
        torch.cuda.amp.autocast(dtype=dtype)
        if autocast_enabled
        else contextlib.nullcontext()
    )

    with torch.no_grad():
        with autocast_context:
            images_batch = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
            depth_maps, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)
    
    depth_map = depth_maps.squeeze().cpu().numpy()
    return depth_map


def load_ground_truth_depth(depth_path):
    """加载 ground truth 深度图"""
    if not os.path.exists(depth_path):
        return None
    
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth is None:
        return None
    
    if len(depth.shape) > 2:
        depth = depth[:, :, 0]
    
    return depth


def colorize_depth(depth_map, cmap='viridis', normalize_individually=True):
    """
    将深度图转换为彩色可视化
    
    Args:
        depth_map: 深度图 numpy 数组
        cmap: 色彩映射名称
        normalize_individually: 是否独立归一化到 0-1（True）还是使用全局范围（False）
    
    Returns:
        colored_rgb: RGB 彩色图像
        vmin, vmax: 使用的深度范围
    """
    if depth_map is None:
        return None, None, None
    
    # 处理 NaN 和 Inf
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 只考虑有效深度值（> 0）
    valid_mask = depth_map > 0
    if not valid_mask.any():
        # 如果没有有效深度，返回黑色图像
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8), 0, 0
    
    # 计算深度范围（每个深度图独立计算）
    vmin = depth_map[valid_mask].min()
    vmax = depth_map[valid_mask].max()
    
    # 归一化到 0-1
    if vmax - vmin < 1e-8:
        normalized = np.zeros_like(depth_map)
    else:
        normalized = np.clip((depth_map - vmin) / (vmax - vmin), 0, 1)
    
    # 应用 colormap (使用新的 API)
    try:
        colormap = plt.colormaps.get_cmap(cmap)
    except AttributeError:
        # 兼容旧版本 matplotlib
        colormap = cm.get_cmap(cmap)
    colored = colormap(normalized)
    
    # 转换为 RGB (0-255)
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    
    # 将无效区域设为黑色
    colored_rgb[~valid_mask] = 0
    
    return colored_rgb, vmin, vmax


def create_comparison_figure(rgb_image, gt_depth, finetuned_depth, baseline_depth, 
                            output_path, scene_name):
    """
    创建对比图：RGB | GT Depth | Fine-tuned Depth | Baseline Depth
    
    关键：每个深度图独立归一化到 0-1，以便更清楚地看到各自的细节
    同时在图像下方显示深度指标（MAE、RMSE等）
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle(f'Model Comparison: {scene_name}', fontsize=16, fontweight='bold', y=0.98)
    
    # 计算指标
    ft_metrics = None
    bl_metrics = None
    if gt_depth is not None:
        if finetuned_depth is not None:
            ft_metrics = compute_metrics(finetuned_depth, gt_depth)
        if baseline_depth is not None:
            bl_metrics = compute_metrics(baseline_depth, gt_depth)
    
    # 1. RGB 图像
    axes[0].imshow(rgb_image)
    axes[0].set_title('RGB Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Ground Truth 深度（独立归一化到 0-1）
    if gt_depth is not None:
        gt_colored, gt_min, gt_max = colorize_depth(gt_depth)
        axes[1].imshow(gt_colored)
        
        # 标题：深度范围
        title_text = f'Ground Truth Depth\nRange: [{gt_min:.2f}, {gt_max:.2f}]'
        axes[1].set_title(title_text, fontsize=11, fontweight='bold')
        
        # 底部：统计信息
        valid_pixels = (gt_depth > 0).sum()
        total_pixels = gt_depth.size
        stats_text = f'Valid: {valid_pixels}/{total_pixels}\n({100*valid_pixels/total_pixels:.1f}%)'
        axes[1].text(0.5, -0.15, stats_text, 
                    transform=axes[1].transAxes,
                    ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        print(f"  GT depth range: [{gt_min:.2f}, {gt_max:.2f}]")
    else:
        axes[1].text(0.5, 0.5, 'No GT Available', ha='center', va='center', fontsize=14)
        axes[1].set_title('Ground Truth Depth', fontsize=11)
    axes[1].axis('off')
    
    # 3. Fine-tuned 模型深度（独立归一化到 0-1）
    if finetuned_depth is not None:
        ft_colored, ft_min, ft_max = colorize_depth(finetuned_depth)
        axes[2].imshow(ft_colored)
        
        # 标题：深度范围
        title_text = f'Fine-tuned Model\nRange: [{ft_min:.2f}, {ft_max:.2f}]'
        axes[2].set_title(title_text, fontsize=11, fontweight='bold', color='green')
        
        # 底部：指标
        if ft_metrics is not None:
            metrics_text = (f'MAE: {ft_metrics["mae"]:.3f}\n'
                          f'RMSE: {ft_metrics["rmse"]:.3f}\n'
                          f'Abs Rel: {ft_metrics["abs_rel"]:.3f}')
            axes[2].text(0.5, -0.15, metrics_text,
                        transform=axes[2].transAxes,
                        ha='center', va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            print(f"  Fine-tuned depth range: [{ft_min:.2f}, {ft_max:.2f}]")
            print(f"  Fine-tuned metrics: MAE={ft_metrics['mae']:.3f}, RMSE={ft_metrics['rmse']:.3f}")
        else:
            axes[2].text(0.5, -0.15, 'No metrics\n(GT unavailable)',
                        transform=axes[2].transAxes,
                        ha='center', va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    else:
        axes[2].text(0.5, 0.5, 'Prediction Failed', ha='center', va='center', fontsize=14)
        axes[2].set_title('Fine-tuned Model', fontsize=11)
    axes[2].axis('off')
    
    # 4. Baseline 模型深度（独立归一化到 0-1）
    if baseline_depth is not None:
        bl_colored, bl_min, bl_max = colorize_depth(baseline_depth)
        axes[3].imshow(bl_colored)
        
        # 标题：深度范围
        title_text = f'Baseline Model\nRange: [{bl_min:.2f}, {bl_max:.2f}]'
        axes[3].set_title(title_text, fontsize=11, fontweight='bold', color='blue')
        
        # 底部：指标
        if bl_metrics is not None:
            metrics_text = (f'MAE: {bl_metrics["mae"]:.3f}\n'
                          f'RMSE: {bl_metrics["rmse"]:.3f}\n'
                          f'Abs Rel: {bl_metrics["abs_rel"]:.3f}')
            axes[3].text(0.5, -0.15, metrics_text,
                        transform=axes[3].transAxes,
                        ha='center', va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            print(f"  Baseline depth range: [{bl_min:.2f}, {bl_max:.2f}]")
            print(f"  Baseline metrics: MAE={bl_metrics['mae']:.3f}, RMSE={bl_metrics['rmse']:.3f}")
        else:
            axes[3].text(0.5, -0.15, 'No metrics\n(GT unavailable)',
                        transform=axes[3].transAxes,
                        ha='center', va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    else:
        axes[3].text(0.5, 0.5, 'Prediction Failed', ha='center', va='center', fontsize=14)
        axes[3].set_title('Baseline Model', fontsize=11)
    axes[3].axis('off')
    
    # 添加改进百分比（如果两个模型都有指标）
    if ft_metrics is not None and bl_metrics is not None:
        improvement_mae = ((bl_metrics['mae'] - ft_metrics['mae']) / bl_metrics['mae']) * 100
        improvement_rmse = ((bl_metrics['rmse'] - ft_metrics['rmse']) / bl_metrics['rmse']) * 100
        
        improvement_text = (f'Improvement:\n'
                          f'MAE: {improvement_mae:+.1f}%\n'
                          f'RMSE: {improvement_rmse:+.1f}%')
        
        # 在图的底部中央显示改进信息
        fig.text(0.5, 0.02, improvement_text,
                ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        print(f"  Improvement: MAE {improvement_mae:+.1f}%, RMSE {improvement_rmse:+.1f}%")
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])  # 为底部文本留出空间
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved comparison: {output_path}")


def compute_metrics(pred_depth, gt_depth):
    """
    计算深度预测指标
    
    Args:
        pred_depth: 预测深度图
        gt_depth: Ground truth 深度图
    
    Returns:
        metrics: 包含各种指标的字典
            - mae: Mean Absolute Error (平均绝对误差)
            - rmse: Root Mean Square Error (均方根误差)
            - abs_rel: Absolute Relative Error (绝对相对误差)
            - sq_rel: Squared Relative Error (平方相对误差)
            - delta1: δ < 1.25 的像素比例
            - delta2: δ < 1.25² 的像素比例
            - delta3: δ < 1.25³ 的像素比例
            - valid_pixels: 有效像素数量
            - total_pixels: 总像素数量
    """
    if pred_depth is None or gt_depth is None:
        return None
    
    # 只在有效区域计算
    valid_mask = (gt_depth > 0) & (pred_depth > 0) & np.isfinite(gt_depth) & np.isfinite(pred_depth)
    if not valid_mask.any():
        return None
    
    pred_valid = pred_depth[valid_mask]
    gt_valid = gt_depth[valid_mask]
    
    # 基本误差指标
    abs_diff = np.abs(pred_valid - gt_valid)
    abs_rel = abs_diff / (gt_valid + 1e-8)
    sq_rel = ((pred_valid - gt_valid) ** 2) / (gt_valid + 1e-8)
    
    # 阈值准确率 (threshold accuracy)
    # δ = max(pred/gt, gt/pred)
    ratio = np.maximum(pred_valid / (gt_valid + 1e-8), gt_valid / (pred_valid + 1e-8))
    delta1 = (ratio < 1.25).mean()
    delta2 = (ratio < 1.25 ** 2).mean()
    delta3 = (ratio < 1.25 ** 3).mean()
    
    metrics = {
        'mae': np.mean(abs_diff),
        'rmse': np.sqrt(np.mean((pred_valid - gt_valid) ** 2)),
        'abs_rel': np.mean(abs_rel),
        'sq_rel': np.mean(sq_rel),
        'delta1': delta1,
        'delta2': delta2,
        'delta3': delta3,
        'valid_pixels': valid_mask.sum(),
        'total_pixels': valid_mask.size
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compare fine-tuned and baseline VGGT models")
    parser.add_argument("--data_dir", type=str, 
                        default="training/dataset_aerialmd_single",
                        help="Path to dataset directory")
    parser.add_argument("--finetuned_model", type=str,
                        default="training/logs/single_pair_test/ckpts/checkpoint.pt",
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--baseline_model", type=str,
                        default="model/vggt_1B_commercial.pt",
                        help="Path to baseline model")
    parser.add_argument("--output", type=str, default="model_comparison",
                        help="Output directory for comparisons")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 设置设备
    if torch.cuda.is_available():
        device = "cuda"
        compute_capability = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if compute_capability[0] >= 8 else torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    
    # 查找数据
    print("\n" + "="*70)
    print("Finding Data")
    print("="*70)
    
    data_dir = Path(args.data_dir)
    
    # 查找场景目录
    scene_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name != 'seg']
    
    if not scene_dirs:
        print(f"❌ No scene directories found in {data_dir}")
        return
    
    print(f"Found {len(scene_dirs)} scene(s): {[d.name for d in scene_dirs]}")
    
    # 收集所有图像信息
    image_data = []
    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        image_extensions = {'.jpg', '.jpeg', '.png'}
        image_files = sorted([f for f in scene_dir.iterdir() 
                            if f.suffix.lower() in image_extensions])
        
        for img_file in image_files:
            img_name = img_file.stem
            depth_file = scene_dir / f"{img_name}.exr"
            
            image_data.append({
                'scene_name': scene_name,
                'img_name': img_name,
                'img_path': str(img_file),
                'depth_path': str(depth_file),
                'rgb_image': None,
                'gt_depth': None,
                'finetuned_depth': None,
                'baseline_depth': None
            })
    
    print(f"Total images to process: {len(image_data)}")
    
    # ========== 阶段 1: 使用 Fine-tuned 模型预测 ==========
    print("\n" + "="*70)
    print("Phase 1: Fine-tuned Model Predictions")
    print("="*70)
    
    try:
        finetuned_model = load_model_from_pt(args.finetuned_model, device)
        
        for i, data in enumerate(image_data):
            print(f"\n[{i+1}/{len(image_data)}] Processing: {data['scene_name']}/{data['img_name']}")
            
            try:
                data['finetuned_depth'] = predict_depth(finetuned_model, data['img_path'], device, dtype)
                print(f"  ✅ Fine-tuned prediction: {data['finetuned_depth'].shape}")
            except Exception as e:
                print(f"  ❌ Fine-tuned prediction failed: {e}")
        
        # 释放模型和显存
        del finetuned_model
        torch.cuda.empty_cache()
        print("\n✅ Fine-tuned model predictions complete. Memory cleared.")
        
    except Exception as e:
        print(f"❌ Failed to load fine-tuned model: {e}")
    
    # ========== 阶段 2: 使用 Baseline 模型预测 ==========
    print("\n" + "="*70)
    print("Phase 2: Baseline Model Predictions")
    print("="*70)
    
    try:
        baseline_model = load_model_from_pt(args.baseline_model, device)
        
        for i, data in enumerate(image_data):
            print(f"\n[{i+1}/{len(image_data)}] Processing: {data['scene_name']}/{data['img_name']}")
            
            try:
                data['baseline_depth'] = predict_depth(baseline_model, data['img_path'], device, dtype)
                print(f"  ✅ Baseline prediction: {data['baseline_depth'].shape}")
            except Exception as e:
                print(f"  ❌ Baseline prediction failed: {e}")
        
        # 释放模型和显存
        del baseline_model
        torch.cuda.empty_cache()
        print("\n✅ Baseline model predictions complete. Memory cleared.")
        
    except Exception as e:
        print(f"❌ Failed to load baseline model: {e}")
    
    # ========== 阶段 3: 加载 GT 和 RGB，生成对比图 ==========
    print("\n" + "="*70)
    print("Phase 3: Creating Comparison Visualizations")
    print("="*70)
    
    all_metrics = {'finetuned': [], 'baseline': []}
    
    for i, data in enumerate(image_data):
        print(f"\n[{i+1}/{len(image_data)}] Creating comparison: {data['scene_name']}/{data['img_name']}")
        
        # 加载 RGB 图像
        data['rgb_image'] = np.array(Image.open(data['img_path']).convert('RGB'))
        
        # 加载 GT 深度
        data['gt_depth'] = load_ground_truth_depth(data['depth_path'])
        if data['gt_depth'] is None:
            print(f"  ⚠️ No ground truth depth found")
        else:
            print(f"  ✅ GT depth loaded: {data['gt_depth'].shape}")
        
        # 创建对比图
        output_path = os.path.join(args.output, 
                                   f"{data['scene_name']}_{data['img_name']}_comparison.png")
        create_comparison_figure(
            data['rgb_image'], data['gt_depth'], 
            data['finetuned_depth'], data['baseline_depth'],
            output_path, f"{data['scene_name']}/{data['img_name']}"
        )
        
        # 计算指标
        if data['gt_depth'] is not None:
            if data['finetuned_depth'] is not None:
                ft_metrics = compute_metrics(data['finetuned_depth'], data['gt_depth'])
                if ft_metrics:
                    all_metrics['finetuned'].append(ft_metrics)
                    print(f"  Fine-tuned MAE: {ft_metrics['mae']:.4f}, RMSE: {ft_metrics['rmse']:.4f}")
            
            if data['baseline_depth'] is not None:
                bl_metrics = compute_metrics(data['baseline_depth'], data['gt_depth'])
                if bl_metrics:
                    all_metrics['baseline'].append(bl_metrics)
                    print(f"  Baseline MAE: {bl_metrics['mae']:.4f}, RMSE: {bl_metrics['rmse']:.4f}")
    
    # 打印总结
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    if all_metrics['finetuned']:
        ft_avg_mae = np.mean([m['mae'] for m in all_metrics['finetuned']])
        ft_avg_rmse = np.mean([m['rmse'] for m in all_metrics['finetuned']])
        print(f"\nFine-tuned Model:")
        print(f"  Average MAE: {ft_avg_mae:.4f}")
        print(f"  Average RMSE: {ft_avg_rmse:.4f}")
    
    if all_metrics['baseline']:
        bl_avg_mae = np.mean([m['mae'] for m in all_metrics['baseline']])
        bl_avg_rmse = np.mean([m['rmse'] for m in all_metrics['baseline']])
        print(f"\nBaseline Model:")
        print(f"  Average MAE: {bl_avg_mae:.4f}")
        print(f"  Average RMSE: {bl_avg_rmse:.4f}")
    
    if all_metrics['finetuned'] and all_metrics['baseline']:
        improvement_mae = ((bl_avg_mae - ft_avg_mae) / bl_avg_mae) * 100
        improvement_rmse = ((bl_avg_rmse - ft_avg_rmse) / bl_avg_rmse) * 100
        print(f"\nImprovement:")
        print(f"  MAE: {improvement_mae:+.2f}%")
        print(f"  RMSE: {improvement_rmse:+.2f}%")
    
    print(f"\n✅ All comparisons saved to: {args.output}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
