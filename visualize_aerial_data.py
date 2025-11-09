#!/usr/bin/env python3
"""
可视化测试脚本：验证 VGGT 中 AerialMegaDepth 数据的正确性
生成可视化图像来检查：
1. RGB 图像
2. Depth Map
3. Segmentation Mask
4. 天空移除效果
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
sys.path.insert(0, str(Path(__file__).parent / "training"))


def visualize_single_sample(dataset, sample_idx=0, save_dir="test_visualizations"):
    """可视化单个样本"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n可视化样本 {sample_idx}...")
    
    # 获取样本信息
    pair = dataset.pairs[sample_idx]
    im1_id = pair['im1_id']
    scene = str(dataset.images_scene_name[im1_id])
    img_name = str(dataset.images[im1_id])
    
    scene_path = os.path.join(dataset.ROOT, scene)
    
    # 加载原始数据
    img_path = os.path.join(scene_path, img_name + '.jpg')
    depth_path = os.path.join(scene_path, img_name + '.exr')
    seg_path = os.path.join(dataset.segmentation_root, scene, img_name + '.png')
    
    # 读取图像
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 读取深度图（原始）
    depth_original = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if len(depth_original.shape) > 2:
        depth_original = depth_original[:, :, 0]
    
    # 读取分割掩码
    segmap = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    
    # 应用分割掩码
    depth_masked = depth_original.copy()
    sky_mask = (segmap == 2)
    depth_masked[sky_mask] = 0
    
    # 通过数据加载器加载（完整处理）
    img_data = dataset._load_image_data(scene, img_name)
    if img_data is not None:
        _, depth_processed, _, _ = img_data
    else:
        depth_processed = depth_masked
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'AerialMegaDepth 数据验证: {scene}/{img_name}', fontsize=16)
    
    # 1. RGB 图像
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('RGB 图像')
    axes[0, 0].axis('off')
    
    # 2. 原始深度图
    depth_vis = depth_original.copy()
    depth_vis[depth_vis == 0] = np.nan
    im1 = axes[0, 1].imshow(depth_vis, cmap='turbo')
    axes[0, 1].set_title(f'原始深度图\n有效像素: {(depth_original > 0).sum()}')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # 3. 分割掩码
    # 创建彩色掩码
    seg_colored = np.zeros((*segmap.shape, 3), dtype=np.uint8)
    seg_colored[segmap == 2] = [135, 206, 235]  # 天空 - 天蓝色
    seg_colored[segmap == 0] = [34, 139, 34]    # 地面 - 绿色
    seg_colored[segmap == 1] = [139, 69, 19]    # 建筑 - 棕色
    # 其他类别用不同颜色
    for label in np.unique(segmap):
        if label not in [0, 1, 2]:
            seg_colored[segmap == label] = np.random.randint(0, 255, 3)
    
    axes[0, 2].imshow(seg_colored)
    axes[0, 2].set_title(f'分割掩码 (ADE20k)\n天空像素: {sky_mask.sum()} ({sky_mask.sum()/segmap.size*100:.1f}%)')
    axes[0, 2].axis('off')
    
    # 4. 天空掩码叠加
    overlay = image.copy()
    overlay[sky_mask] = overlay[sky_mask] * 0.3 + np.array([255, 0, 0]) * 0.7
    axes[1, 0].imshow(overlay.astype(np.uint8))
    axes[1, 0].set_title('天空区域标注 (红色)')
    axes[1, 0].axis('off')
    
    # 5. 应用掩码后的深度图
    depth_masked_vis = depth_masked.copy()
    depth_masked_vis[depth_masked_vis == 0] = np.nan
    im2 = axes[1, 1].imshow(depth_masked_vis, cmap='turbo')
    axes[1, 1].set_title(f'天空移除后深度图\n有效像素: {(depth_masked > 0).sum()}')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    
    # 6. 最终处理后的深度图
    depth_processed_vis = depth_processed.copy()
    depth_processed_vis[depth_processed_vis == 0] = np.nan
    im3 = axes[1, 2].imshow(depth_processed_vis, cmap='turbo')
    axes[1, 2].set_title(f'完整处理后深度图\n有效像素: {(depth_processed > 0).sum()}')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, f'sample_{sample_idx}_{scene}_{img_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化已保存: {save_path}")
    
    # 显示统计信息
    print(f"\n统计信息:")
    print(f"  - 图像尺寸: {image.shape}")
    print(f"  - 原始有效深度: {(depth_original > 0).sum()} / {depth_original.size}")
    print(f"  - 天空像素: {sky_mask.sum()} ({sky_mask.sum()/segmap.size*100:.1f}%)")
    print(f"  - 天空区域有效深度: {(sky_mask & (depth_original > 0)).sum()}")
    print(f"  - 掩码后有效深度: {(depth_masked > 0).sum()}")
    print(f"  - 最终有效深度: {(depth_processed > 0).sum()}")
    print(f"  - 移除像素: {(depth_original > 0).sum() - (depth_processed > 0).sum()}")
    
    plt.close()
    
    return True


def visualize_batch(dataset, save_dir="test_visualizations"):
    """可视化一个批次"""
    
    print(f"\n可视化批次数据...")
    
    # 生成批次
    batch = dataset.get_data(seq_index=0, img_per_seq=2, aspect_ratio=1.0)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'批次数据: {batch["seq_name"]}', fontsize=16)
    
    for i in range(min(2, batch['frame_num'])):
        # RGB
        axes[i, 0].imshow(batch['images'][i])
        axes[i, 0].set_title(f'图像 {i}')
        axes[i, 0].axis('off')
        
        # Depth
        depth = batch['depths'][i].copy()
        depth[depth == 0] = np.nan
        im = axes[i, 1].imshow(depth, cmap='turbo')
        axes[i, 1].set_title(f'深度图 {i}')
        axes[i, 1].axis('off')
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046)
        
        # Point mask
        axes[i, 2].imshow(batch['point_masks'][i], cmap='gray')
        axes[i, 2].set_title(f'有效点掩码 {i}')
        axes[i, 2].axis('off')
        
        # Camera points (depth visualization)
        cam_points = batch['cam_points'][i]
        cam_depth = np.linalg.norm(cam_points, axis=-1)
        cam_depth[cam_depth == 0] = np.nan
        im2 = axes[i, 3].imshow(cam_depth, cmap='turbo')
        axes[i, 3].set_title(f'相机坐标深度 {i}')
        axes[i, 3].axis('off')
        plt.colorbar(im2, ax=axes[i, 3], fraction=0.046)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'batch_{batch["seq_name"]}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 批次可视化已保存: {save_path}")
    
    plt.close()
    
    return True


def main():
    """主函数"""
    print("="*70)
    print("VGGT AerialMegaDepth 数据可视化测试")
    print("="*70)
    
    # 初始化数据集
    from data.datasets.megadepth_aerial import MegaDepthAerialDataset
    from types import SimpleNamespace
    
    common_conf = SimpleNamespace(
        img_size=518,
        patch_size=14,
        debug=False,
        training=True,
        get_nearby=False,
        inside_random=False,
        allow_duplicate_img=False,
        repeat_batch=False,
        rescale=True,
        rescale_aug=True,
        landscape_check=True,
        augs=SimpleNamespace(
            scales=[1.0]
        )
    )
    
    dataset = MegaDepthAerialDataset(
        common_conf=common_conf,
        split="train",
        ROOT="/home/haowei/Documents/vggt/training/dataset_aerialmd/cropped",
        split_file="train.npz",
        segmentation_root="/home/haowei/Documents/vggt/training/dataset_aerialmd/cropped_seg",
        max_depth=2000.0,
        depth_percentile=98.0,
        use_pairs=True,
        expand_ratio=2,
        remove_sky=True,
    )
    
    print(f"✅ 数据集加载成功")
    print(f"   - 场景: {dataset.valid_scenes}")
    print(f"   - 配对数: {len(dataset.pairs)}")
    
    # 创建输出目录
    save_dir = "test_visualizations"
    os.makedirs(save_dir, exist_ok=True)
    
    # 可视化多个样本
    num_samples = min(3, len(dataset.pairs))
    for i in range(num_samples):
        visualize_single_sample(dataset, sample_idx=i, save_dir=save_dir)
    
    # 可视化批次
    visualize_batch(dataset, save_dir=save_dir)
    
    print("\n" + "="*70)
    print(f"✅ 所有可视化完成！结果保存在: {save_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()
