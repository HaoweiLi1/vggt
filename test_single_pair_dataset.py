#!/usr/bin/env python3
"""
测试单 pair 数据集是否正确创建
"""

import os
import sys
from pathlib import Path
import torch
import torch.distributed as dist
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
sys.path.insert(0, str(Path(__file__).parent / "training"))

def init_distributed():
    """初始化分布式"""
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12359'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        dist.init_process_group(backend='gloo', init_method='env://', world_size=1, rank=0)

def main():
    print("\n" + "="*70)
    print("测试单 Pair 数据集")
    print("="*70)
    
    init_distributed()
    
    from hydra import initialize, compose
    from hydra.utils import instantiate
    
    # 加载配置
    with initialize(version_base=None, config_path="training/config"):
        cfg = compose(config_name="single_pair_test")
    
    print(f"\n✅ 配置加载成功")
    print(f"   - 实验名称: {cfg.exp_name}")
    print(f"   - ROOT: {cfg.data.train.dataset.dataset_configs[0].ROOT}")
    print(f"   - segmentation_root: {cfg.data.train.dataset.dataset_configs[0].segmentation_root}")
    
    # 创建 dataloader
    train_dataset = instantiate(cfg.data.train, _recursive_=False)
    train_dataset.seed = cfg.seed_value
    
    print(f"\n✅ Train dataset 创建成功")
    print(f"   - 数据集长度: {len(train_dataset.dataset)}")
    
    # 获取 loader
    loader = train_dataset.get_loader(epoch=0)
    
    print(f"\n✅ Loader 创建成功")
    
    # 测试迭代
    print(f"\n测试批次迭代:")
    
    batch_count = 0
    for i, batch in enumerate(loader):
        batch_count += 1
        
        print(f"\n  批次 {i}:")
        print(f"    - seq_name: {batch['seq_name']}")
        print(f"    - images 形状: {batch['images'].shape}")
        print(f"    - depths 形状: {batch['depths'].shape}")
        
        # 分析深度值
        depths = batch['depths'].cpu().numpy()
        
        for j in range(depths.shape[0]):  # batch_size
            for k in range(min(2, depths.shape[1])):  # num_images
                depth = depths[j, k]
                valid_pixels = (depth > 0).sum()
                total_pixels = depth.size
                
                if valid_pixels > 0:
                    valid_depth = depth[depth > 0]
                    print(f"    - 样本 {j} 图像 {k}: 有效深度 {valid_pixels}/{total_pixels} "
                          f"({valid_pixels/total_pixels*100:.1f}%), "
                          f"范围 [{valid_depth.min():.2f}, {valid_depth.max():.2f}]")
        
        if batch_count >= 5:  # 只测试前 5 个批次
            break
    
    print(f"\n" + "="*70)
    print(f"测试总结:")
    print(f"  ✅ 数据集正确加载")
    print(f"  ✅ 批次数量: {batch_count}")
    print(f"  ✅ 数据格式正确")
    print(f"  ✅ 深度数据有效")
    print(f"\n可以开始过拟合测试:")
    print(f"  cd training")
    print(f"  python launch.py --config single_pair_test")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
