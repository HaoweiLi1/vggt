#!/usr/bin/env python3
"""
使用 training 中真实的 dataloader 测试 AerialMegaDepth 数据加载
这个脚本直接使用 training/config/default.yaml 中的配置来初始化 dataloader
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
import torch.distributed as dist

# 设置环境变量
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# 添加训练目录到路径
sys.path.insert(0, str(Path(__file__).parent / "training"))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def init_distributed_mode():
    """初始化分布式模式（单GPU测试）"""
    if not dist.is_initialized():
        # 设置环境变量
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        
        # 初始化进程组
        dist.init_process_group(
            backend='gloo',  # 使用 gloo 后端（CPU）
            init_method='env://',
            world_size=1,
            rank=0
        )
        print("✅ 分布式进程组已初始化（单GPU模式）")

def test_with_hydra_config():
    """使用 Hydra 配置测试"""
    print("\n" + "="*70)
    print("测试 1: 使用 Hydra 配置加载 dataloader")
    print("="*70)
    
    try:
        from hydra import initialize, compose
        from hydra.utils import instantiate
        
        # 初始化 Hydra
        with initialize(version_base=None, config_path="training/config"):
            # 加载配置
            cfg = compose(config_name="default")
            
            print(f"✅ 配置加载成功")
            print(f"   - 实验名称: {cfg.exp_name}")
            print(f"   - 图像尺寸: {cfg.img_size}")
            print(f"   - 数据集配置:")
            print(f"     ROOT: {cfg.data.train.dataset.dataset_configs[0].ROOT}")
            print(f"     split_file: {cfg.data.train.dataset.dataset_configs[0].split_file}")
            print(f"     segmentation_root: {cfg.data.train.dataset.dataset_configs[0].segmentation_root}")
            print(f"     remove_sky: {cfg.data.train.dataset.dataset_configs[0].get('remove_sky', True)}")
            
            # 实例化 dataloader
            print(f"\n   正在实例化 train dataloader...")
            train_dataloader = instantiate(cfg.data.train, _recursive_=False)
            
            print(f"✅ Train dataloader 创建成功")
            print(f"   - 数据集类型: {type(train_dataloader.dataset)}")
            print(f"   - 数据集长度: {len(train_dataloader.dataset)}")
            
            # 实例化 val dataloader
            print(f"\n   正在实例化 val dataloader...")
            val_dataloader = instantiate(cfg.data.val, _recursive_=False)
            
            print(f"✅ Val dataloader 创建成功")
            print(f"   - 数据集类型: {type(val_dataloader.dataset)}")
            print(f"   - 数据集长度: {len(val_dataloader.dataset)}")
            
            return train_dataloader, val_dataloader, cfg, True
            
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, False


def test_dataloader_iteration(dataloader, split_name="train", num_batches=3):
    """测试 dataloader 迭代"""
    print("\n" + "="*70)
    print(f"测试 2: {split_name} dataloader 迭代测试")
    print("="*70)
    
    try:
        # 获取 PyTorch DataLoader
        loader = dataloader.get_loader(epoch=0)
        
        print(f"✅ DataLoader 创建成功")
        print(f"   - Batch sampler: {type(loader.batch_sampler)}")
        print(f"   - Num workers: {loader.num_workers}")
        
        # 迭代几个批次
        success_count = 0
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
                
            try:
                print(f"\n   批次 {i}:")
                
                # Training dataloader 返回的是列表格式
                if isinstance(batch, list):
                    print(f"     - 批次大小: {len(batch)} 个样本")
                    
                    for sample_idx, sample in enumerate(batch):
                        print(f"     样本 {sample_idx}:")
                        print(f"       - seq_name: {sample['seq_name']}")
                        print(f"       - frame_num: {sample['frame_num']}")
                        print(f"       - images 形状: {[img.shape for img in sample['images']]}")
                        print(f"       - depths 形状: {[d.shape for d in sample['depths']]}")
                        
                        # 检查深度值
                        for j, depth in enumerate(sample['depths']):
                            valid_depth = depth[depth > 0]
                            if len(valid_depth) > 0:
                                print(f"       - 图像 {j} 有效深度: {len(valid_depth)} 像素, "
                                      f"范围: [{valid_depth.min():.2f}, {valid_depth.max():.2f}]")
                            else:
                                print(f"       - 图像 {j}: ⚠️ 没有有效深度")
                else:
                    # 单个样本格式
                    print(f"     - seq_name: {batch['seq_name']}")
                    print(f"     - frame_num: {batch['frame_num']}")
                    print(f"     - images 形状: {[img.shape for img in batch['images']]}")
                
                success_count += 1
                
            except Exception as e:
                print(f"     ❌ 批次 {i} 处理失败: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n   总结: {success_count}/{num_batches} 批次成功")
        return success_count == num_batches
        
    except Exception as e:
        print(f"❌ DataLoader 迭代失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_segmentation_in_batch(dataloader):
    """测试批次中的分割掩码效果"""
    print("\n" + "="*70)
    print("测试 3: 验证分割掩码在批次中的应用")
    print("="*70)
    
    try:
        loader = dataloader.get_loader(epoch=0)
        
        # 获取一个批次
        batch = next(iter(loader))
        
        print(f"✅ 获取批次成功")
        
        # Training dataloader 返回列表格式
        if isinstance(batch, list):
            print(f"   - 批次大小: {len(batch)} 个样本")
            sample = batch[0]  # 取第一个样本分析
        else:
            sample = batch
        
        print(f"   - 序列: {sample['seq_name']}")
        print(f"   - 图像数: {sample['frame_num']}")
        
        # 分析深度图
        for i, depth in enumerate(sample['depths']):
            depth_np = depth.cpu().numpy() if hasattr(depth, 'cpu') else depth
            
            total_pixels = depth_np.size
            valid_pixels = (depth_np > 0).sum()
            zero_pixels = (depth_np == 0).sum()
            
            print(f"\n   图像 {i}:")
            print(f"     - 总像素: {total_pixels}")
            print(f"     - 有效深度: {valid_pixels} ({valid_pixels/total_pixels*100:.1f}%)")
            print(f"     - 零值像素: {zero_pixels} ({zero_pixels/total_pixels*100:.1f}%)")
            
            if valid_pixels > 0:
                valid_depth = depth_np[depth_np > 0]
                print(f"     - 深度范围: [{valid_depth.min():.2f}, {valid_depth.max():.2f}]")
                print(f"     - 深度均值: {valid_depth.mean():.2f}")
        
        # 检查零值像素比例是否合理（应该包含天空区域）
        avg_zero_ratio = np.mean([((d > 0).sum() / d.size) for d in sample['depths']])
        
        if 0.5 < avg_zero_ratio < 0.9:
            print(f"\n   ✅ 零值像素比例合理 (平均有效像素: {avg_zero_ratio*100:.1f}%)")
            print(f"      这表明分割掩码可能已正确应用（天空区域被移除）")
            return True
        else:
            print(f"\n   ⚠️ 零值像素比例异常 (平均有效像素: {avg_zero_ratio*100:.1f}%)")
            return False
            
    except Exception as e:
        print(f"❌ 分割掩码验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_batch(dataloader, save_dir="test_training_visualizations"):
    """可视化一个批次"""
    print("\n" + "="*70)
    print("测试 4: 可视化批次数据")
    print("="*70)
    
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        loader = dataloader.get_loader(epoch=0)
        batch = next(iter(loader))
        
        # Training dataloader 返回列表格式
        if isinstance(batch, list):
            sample = batch[0]  # 取第一个样本
        else:
            sample = batch
        
        num_images = min(2, sample['frame_num'])
        
        fig, axes = plt.subplots(num_images, 3, figsize=(15, 5*num_images))
        if num_images == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Training Dataloader 批次: {sample["seq_name"]}', fontsize=14)
        
        for i in range(num_images):
            # RGB
            img = sample['images'][i]
            if hasattr(img, 'cpu'):
                img = img.cpu().numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'RGB 图像 {i}')
            axes[i, 0].axis('off')
            
            # Depth
            depth = sample['depths'][i]
            if hasattr(depth, 'cpu'):
                depth = depth.cpu().numpy()
            depth_vis = depth.copy()
            depth_vis[depth_vis == 0] = np.nan
            im = axes[i, 1].imshow(depth_vis, cmap='turbo')
            axes[i, 1].set_title(f'深度图 {i}\n有效: {(depth > 0).sum()}/{depth.size}')
            axes[i, 1].axis('off')
            plt.colorbar(im, ax=axes[i, 1], fraction=0.046)
            
            # Point mask
            if 'point_masks' in sample:
                mask = sample['point_masks'][i]
                if hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy()
                axes[i, 2].imshow(mask, cmap='gray')
                axes[i, 2].set_title(f'有效点掩码 {i}')
                axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'training_batch_{sample["seq_name"]}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 可视化已保存: {save_path}")
        
        plt.close()
        return True
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_with_manual_config():
    """对比手动配置和 Hydra 配置"""
    print("\n" + "="*70)
    print("测试 5: 对比手动配置 vs Hydra 配置")
    print("="*70)
    
    try:
        from hydra import initialize, compose
        from hydra.utils import instantiate
        from types import SimpleNamespace
        from data.datasets.megadepth_aerial import MegaDepthAerialDataset
        
        # 1. Hydra 配置
        with initialize(version_base=None, config_path="training/config"):
            cfg = compose(config_name="default")
            hydra_dataloader = instantiate(cfg.data.train, _recursive_=False)
            hydra_dataset = hydra_dataloader.dataset
        
        # 2. 手动配置（之前的测试脚本方式）
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
            augs=SimpleNamespace(scales=[1.0])
        )
        
        manual_dataset = MegaDepthAerialDataset(
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
        
        print(f"✅ 两种配置都成功创建")
        print(f"\n   Hydra 配置:")
        print(f"     - 数据集类型: {type(hydra_dataset)}")
        print(f"     - 数据集长度: {len(hydra_dataset)}")
        print(f"     - ROOT: {hydra_dataset.datasets[0].ROOT if hasattr(hydra_dataset, 'datasets') else 'N/A'}")
        
        print(f"\n   手动配置:")
        print(f"     - 数据集类型: {type(manual_dataset)}")
        print(f"     - 数据集长度: {len(manual_dataset)}")
        print(f"     - ROOT: {manual_dataset.ROOT}")
        print(f"     - segmentation_root: {manual_dataset.segmentation_root}")
        print(f"     - remove_sky: {manual_dataset.remove_sky}")
        
        # 对比关键参数
        print(f"\n   关键参数对比:")
        
        # 获取 Hydra 配置的实际数据集
        actual_dataset = hydra_dataset.datasets[0] if hasattr(hydra_dataset, 'datasets') else hydra_dataset
        
        params_match = True
        if hasattr(actual_dataset, 'segmentation_root'):
            if actual_dataset.segmentation_root == manual_dataset.segmentation_root:
                print(f"     ✅ segmentation_root 一致")
            else:
                print(f"     ❌ segmentation_root 不一致")
                print(f"        Hydra: {actual_dataset.segmentation_root}")
                print(f"        手动: {manual_dataset.segmentation_root}")
                params_match = False
        
        if hasattr(actual_dataset, 'remove_sky'):
            if actual_dataset.remove_sky == manual_dataset.remove_sky:
                print(f"     ✅ remove_sky 一致")
            else:
                print(f"     ❌ remove_sky 不一致")
                params_match = False
        
        if hasattr(actual_dataset, 'max_depth'):
            if actual_dataset.max_depth == manual_dataset.max_depth:
                print(f"     ✅ max_depth 一致")
            else:
                print(f"     ❌ max_depth 不一致")
                params_match = False
        
        return params_match
        
    except Exception as e:
        print(f"❌ 对比失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*70)
    print("使用 Training 真实 Dataloader 测试 AerialMegaDepth")
    print("="*70)
    
    # 初始化分布式模式
    init_distributed_mode()
    
    results = {}
    
    # 测试 1: 使用 Hydra 配置
    train_dataloader, val_dataloader, cfg, success = test_with_hydra_config()
    results['Hydra 配置加载'] = success
    
    if not success:
        print("\n❌ 无法加载配置，测试终止")
        return
    
    # 测试 2: Train dataloader 迭代
    results['Train Dataloader 迭代'] = test_dataloader_iteration(train_dataloader, "train", num_batches=3)
    
    # 测试 3: Val dataloader 迭代
    results['Val Dataloader 迭代'] = test_dataloader_iteration(val_dataloader, "val", num_batches=2)
    
    # 测试 4: 分割掩码验证
    results['分割掩码验证'] = test_segmentation_in_batch(train_dataloader)
    
    # 测试 5: 可视化
    results['批次可视化'] = visualize_batch(train_dataloader)
    
    # 测试 6: 配置对比
    results['配置对比'] = compare_with_manual_config()
    
    # 最终总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\n   总计: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！Training dataloader 正确使用了 AerialMegaDepth 数据集！")
    else:
        print(f"\n⚠️ 有 {total_tests - passed_tests} 个测试失败")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
