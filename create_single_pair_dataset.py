#!/usr/bin/env python3
"""
从完整的 AerialMegaDepth 数据集中提取单个 pair，创建用于过拟合测试的数据集
目的：验证模型能否在单个样本上将 loss 降到 0

输出目录结构：
dataset_aerialmd_single/
├── train.npz                    # 只包含 1 个 pair
├── val.npz                      # 只包含 1 个 pair（相同的）
├── 0001/                        # 场景目录
│   ├── image1.jpg.jpg          # RGB 图像 1
│   ├── image1.jpg.exr          # Depth map 1
│   ├── image1.jpg.npz          # Camera params 1
│   ├── image2.jpg.jpg          # RGB 图像 2
│   ├── image2.jpg.exr          # Depth map 2
│   └── image2.jpg.npz          # Camera params 2
└── 0001_seg/                    # 分割掩码目录
    ├── image1.jpg.png          # Segmentation mask 1
    └── image2.jpg.png          # Segmentation mask 2
"""

import os
import sys
import shutil
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_npz_data(npz_path):
    """加载 NPZ 文件"""
    with np.load(npz_path, allow_pickle=True) as data:
        return {
            'scenes': data['scenes'],
            'images': data['images'],
            'pairs': data['pairs'],
            'images_scene_name': data.get('images_scene_name', None)
        }

def create_single_pair_dataset(
    source_root="/home/haowei/Documents/vggt/training/dataset_aerialmd/cropped",
    source_seg_root="/home/haowei/Documents/vggt/training/dataset_aerialmd/cropped_seg",
    source_npz="train.npz",
    output_root="/home/haowei/Documents/vggt/training/dataset_aerialmd_single",
    pair_index=0
):
    """
    从源数据集中提取单个 pair 创建新数据集
    
    Args:
        source_root: 源数据集根目录
        source_seg_root: 源分割掩码根目录
        source_npz: 源 NPZ 文件名
        output_root: 输出数据集根目录
        pair_index: 要提取的 pair 索引
    """
    
    print("="*70)
    print("创建单 Pair 数据集用于过拟合测试")
    print("="*70)
    
    # 创建输出目录
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    output_seg_root = output_root / "seg"
    output_seg_root.mkdir(parents=True, exist_ok=True)
    
    # 加载源 NPZ
    source_npz_path = Path(source_root) / source_npz
    print(f"\n步骤 1: 加载源 NPZ 文件")
    print(f"  - 路径: {source_npz_path}")
    
    data = load_npz_data(source_npz_path)
    
    print(f"  ✅ 加载成功")
    print(f"     - 场景数: {len(data['scenes'])}")
    print(f"     - 图像数: {len(data['images'])}")
    print(f"     - 配对数: {len(data['pairs'])}")
    
    # 选择一个 pair
    print(f"\n步骤 2: 选择 pair {pair_index}")
    
    if pair_index >= len(data['pairs']):
        print(f"  ❌ pair_index {pair_index} 超出范围，使用 pair 0")
        pair_index = 0
    
    pair = data['pairs'][pair_index]
    scene_id = int(pair[0])
    im1_id = int(pair[1])
    im2_id = int(pair[2])
    score = float(pair[3]) if len(pair) > 3 else 1.0
    
    scene_name = str(data['scenes'][scene_id])
    im1_name = str(data['images'][im1_id])
    im2_name = str(data['images'][im2_id])
    
    # 获取实际场景名称
    if data['images_scene_name'] is not None:
        scene_im1 = str(data['images_scene_name'][im1_id])
        scene_im2 = str(data['images_scene_name'][im2_id])
        
        if scene_im1 != scene_im2:
            print(f"  ⚠️ 警告: 两张图像来自不同场景")
            print(f"     - 图像 1: {scene_im1}")
            print(f"     - 图像 2: {scene_im2}")
            scene_name = scene_im1
        else:
            scene_name = scene_im1
    
    print(f"  ✅ 选择的 pair:")
    print(f"     - 场景: {scene_name}")
    print(f"     - 图像 1: {im1_name}")
    print(f"     - 图像 2: {im2_name}")
    print(f"     - 分数: {score}")
    
    # 复制场景目录
    print(f"\n步骤 3: 复制数据文件")
    
    source_scene_dir = Path(source_root) / scene_name
    output_scene_dir = output_root / scene_name
    output_scene_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制图像 1 的所有文件
    files_to_copy = [
        (im1_name + '.jpg', 'RGB 图像 1'),
        (im1_name + '.exr', 'Depth map 1'),
        (im1_name + '.npz', 'Camera params 1'),
        (im2_name + '.jpg', 'RGB 图像 2'),
        (im2_name + '.exr', 'Depth map 2'),
        (im2_name + '.npz', 'Camera params 2'),
    ]
    
    copied_files = []
    for filename, description in files_to_copy:
        source_file = source_scene_dir / filename
        output_file = output_scene_dir / filename
        
        if source_file.exists():
            shutil.copy2(source_file, output_file)
            print(f"  ✅ {description}: {filename}")
            copied_files.append(filename)
        else:
            print(f"  ❌ {description}: {filename} (不存在)")
    
    # 复制分割掩码
    print(f"\n步骤 4: 复制分割掩码")
    
    source_seg_scene_dir = Path(source_seg_root) / scene_name
    output_seg_scene_dir = output_seg_root / scene_name
    output_seg_scene_dir.mkdir(parents=True, exist_ok=True)
    
    seg_files_to_copy = [
        (im1_name + '.png', 'Segmentation mask 1'),
        (im2_name + '.png', 'Segmentation mask 2'),
    ]
    
    for filename, description in seg_files_to_copy:
        source_file = source_seg_scene_dir / filename
        output_file = output_seg_scene_dir / filename
        
        if source_file.exists():
            shutil.copy2(source_file, output_file)
            print(f"  ✅ {description}: {filename}")
        else:
            print(f"  ⚠️ {description}: {filename} (不存在，跳过)")
    
    # 创建新的 NPZ 文件
    print(f"\n步骤 5: 创建新的 NPZ 文件")
    
    # 新的数据结构
    new_scenes = np.array([scene_name], dtype=object)
    new_images = np.array([im1_name, im2_name], dtype=object)
    new_pairs = np.array([[0, 0, 1, score]], dtype=object)  # scene_id=0, im1_id=0, im2_id=1
    new_images_scene_name = np.array([scene_name, scene_name], dtype=object)
    
    # 保存 train.npz
    train_npz_path = output_root / 'train.npz'
    np.savez(
        train_npz_path,
        scenes=new_scenes,
        images=new_images,
        pairs=new_pairs,
        images_scene_name=new_images_scene_name
    )
    print(f"  ✅ train.npz 创建成功")
    print(f"     - 场景数: {len(new_scenes)}")
    print(f"     - 图像数: {len(new_images)}")
    print(f"     - 配对数: {len(new_pairs)}")
    
    # 保存 val.npz (相同的数据)
    val_npz_path = output_root / 'val.npz'
    np.savez(
        val_npz_path,
        scenes=new_scenes,
        images=new_images,
        pairs=new_pairs,
        images_scene_name=new_images_scene_name
    )
    print(f"  ✅ val.npz 创建成功 (与 train.npz 相同)")
    
    # 验证数据完整性
    print(f"\n步骤 6: 验证数据完整性")
    
    required_files = [
        (output_scene_dir / (im1_name + '.jpg'), 'RGB 1'),
        (output_scene_dir / (im1_name + '.exr'), 'Depth 1'),
        (output_scene_dir / (im1_name + '.npz'), 'Cam 1'),
        (output_scene_dir / (im2_name + '.jpg'), 'RGB 2'),
        (output_scene_dir / (im2_name + '.exr'), 'Depth 2'),
        (output_scene_dir / (im2_name + '.npz'), 'Cam 2'),
    ]
    
    all_exist = True
    for file_path, description in required_files:
        if file_path.exists():
            size = file_path.stat().st_size / 1024  # KB
            print(f"  ✅ {description}: {file_path.name} ({size:.1f} KB)")
        else:
            print(f"  ❌ {description}: {file_path.name} (缺失)")
            all_exist = False
    
    # 检查分割掩码
    seg_files = [
        (output_seg_scene_dir / (im1_name + '.png'), 'Seg 1'),
        (output_seg_scene_dir / (im2_name + '.png'), 'Seg 2'),
    ]
    
    for file_path, description in seg_files:
        if file_path.exists():
            size = file_path.stat().st_size / 1024  # KB
            print(f"  ✅ {description}: {file_path.name} ({size:.1f} KB)")
        else:
            print(f"  ⚠️ {description}: {file_path.name} (缺失)")
    
    # 最终总结
    print(f"\n" + "="*70)
    print("数据集创建完成")
    print("="*70)
    print(f"输出目录: {output_root}")
    print(f"\n目录结构:")
    print(f"  {output_root}/")
    print(f"  ├── train.npz                 # 1 个 pair")
    print(f"  ├── val.npz                   # 1 个 pair (相同)")
    print(f"  ├── {scene_name}/")
    print(f"  │   ├── {im1_name}.jpg        # RGB 1")
    print(f"  │   ├── {im1_name}.exr        # Depth 1")
    print(f"  │   ├── {im1_name}.npz        # Cam 1")
    print(f"  │   ├── {im2_name}.jpg        # RGB 2")
    print(f"  │   ├── {im2_name}.exr        # Depth 2")
    print(f"  │   └── {im2_name}.npz        # Cam 2")
    print(f"  └── seg/")
    print(f"      └── {scene_name}/")
    print(f"          ├── {im1_name}.png    # Seg 1")
    print(f"          └── {im2_name}.png    # Seg 2")
    
    if all_exist:
        print(f"\n✅ 所有必需文件都已创建")
        print(f"\n下一步:")
        print(f"  1. 更新配置文件使用新数据集:")
        print(f"     ROOT: {output_root}")
        print(f"     segmentation_root: {output_seg_root}")
        print(f"  2. 运行训练，观察 loss 是否能降到 0")
        print(f"     cd training")
        print(f"     python launch.py --config single_pair_test")
    else:
        print(f"\n⚠️ 部分文件缺失，请检查源数据集")
    
    print("="*70 + "\n")
    
    return output_root


def create_config_for_single_pair(output_root):
    """创建用于单 pair 测试的配置文件"""
    
    config_content = f"""# 单 Pair 过拟合测试配置
# 目的：验证模型能否在单个样本上将 loss 降到 0

defaults:
  - default_dataset.yaml

exp_name: single_pair_overfit_test
img_size: 518
num_workers: 0  # 单线程，便于调试
seed_value: 42
accum_steps: 1
patch_size: 14
val_epoch_freq: 1  # 每个 epoch 都验证
max_img_per_gpu: 2

# 不限制批次数量，使用所有数据（只有 1 个 pair）
limit_train_batches: null
limit_val_batches: null

data:
  train:
    _target_: data.dynamic_dataloader.DynamicTorchDataset
    num_workers: ${{num_workers}}
    max_img_per_gpu: ${{max_img_per_gpu}}
    common_config:
      img_size: ${{img_size}}
      patch_size: ${{patch_size}}
      debug: True  # 启用调试模式
      repeat_batch: False
    dataset:
      _target_: data.composed_dataset.ComposedDataset
      dataset_configs:
        - _target_: data.datasets.megadepth_aerial.MegaDepthAerialDataset
          split: train
          ROOT: {output_root}
          split_file: train.npz
          segmentation_root: {output_root}/seg
          remove_sky: true
          max_depth: 2000.0
          depth_percentile: 98.0
          use_pairs: true
          expand_ratio: 2
          len_train: 1  # 只有 1 个 pair
      
  val:
    _target_: data.dynamic_dataloader.DynamicTorchDataset
    num_workers: ${{num_workers}}
    max_img_per_gpu: ${{max_img_per_gpu}}
    common_config:
      img_size: ${{img_size}}
      patch_size: ${{patch_size}}
      debug: True
      repeat_batch: False
    dataset:
      _target_: data.composed_dataset.ComposedDataset
      dataset_configs:
        - _target_: data.datasets.megadepth_aerial.MegaDepthAerialDataset
          split: val
          ROOT: {output_root}
          split_file: val.npz
          segmentation_root: {output_root}/seg
          remove_sky: true
          max_depth: 2000.0
          depth_percentile: 98.0
          use_pairs: true
          expand_ratio: 2
          len_test: 1  # 只有 1 个 pair

logging:
  log_dir: logs/single_pair_test
  log_visuals: True  # 启用可视化
  log_freq: 1  # 每个 batch 都记录
  log_level_primary: DEBUG
  log_level_secondary: INFO
  all_ranks: False
  tensorboard_writer:
    _target_: train_utils.tb_writer.TensorBoardLogger
    path: ${{logging.log_dir}}/tensorboard
  scalar_keys_to_log:
    train:
      keys_to_log:
        - loss_objective
        - loss_camera
        - loss_T
        - loss_R
        - loss_FL
        - loss_conf_depth
        - loss_reg_depth
        - loss_grad_depth
    val:
      keys_to_log:
        - loss_objective
        - loss_camera
        - loss_T
        - loss_R
        - loss_FL
        - loss_conf_depth
        - loss_reg_depth
        - loss_grad_depth

checkpoint:
  save_dir: logs/single_pair_test/ckpts
  save_freq: 10  # 每 10 个 epoch 保存一次
  resume_checkpoint_path: /home/haowei/Documents/vggt/model/vggt_1B_commercial.pt
  strict: False

loss:
  _target_: loss.MultitaskLoss
  camera: 
    weight: 5.0
    loss_type: "l1"
  depth:
    weight: 1.0
    gradient_loss_fn: "grad" 
    valid_range: 0.98
  point: null
  track: null

optim:
  param_group_modifiers: False

  optimizer:
    _target_: torch.optim.AdamW
    lr: 1e-4  # 较高的学习率，加快过拟合
    weight_decay: 0.01  # 较小的权重衰减

  frozen_module_names: []  # 不冻结任何模块

  amp:
    enabled: True
    amp_dtype: bfloat16
    
  gradient_clip:
    _target_: train_utils.gradient_clip.GradientClipper
    configs:
      - module_name: ["aggregator"]
        max_norm: 1.0
        norm_type: 2
      - module_name: ["depth"]
        max_norm: 1.0
        norm_type: 2
      - module_name: ["camera"]
        max_norm: 1.0
        norm_type: 2
        
  options:
    lr:
      - scheduler:
          _target_: fvcore.common.param_scheduler.ConstantParamScheduler
          value: 1e-4  # 固定学习率
    weight_decay:
      - scheduler:
          _target_: fvcore.common.param_scheduler.ConstantParamScheduler
          value: 0.01

max_epochs: 100  # 训练 100 个 epoch，观察 loss 下降

model:
  _target_: vggt.models.vggt.VGGT
  enable_camera: True
  enable_depth: True
  enable_point: False
  enable_track: False
  use_vit_features: True

distributed:
  backend: nccl
  comms_dtype: None
  find_unused_parameters: False
  timeout_mins: 30
  gradient_as_bucket_view: True
  bucket_cap_mb: 25
  broadcast_buffers: True

cuda:
    cudnn_deterministic: False
    cudnn_benchmark: False
    allow_tf32: True
"""
    
    config_path = Path("training/config/single_pair_test.yaml")
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"\n✅ 配置文件已创建: {config_path}")
    return config_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="创建单 pair 数据集用于过拟合测试")
    parser.add_argument(
        '--source_root',
        type=str,
        default="/home/haowei/Documents/vggt/training/dataset_aerialmd/cropped",
        help="源数据集根目录"
    )
    parser.add_argument(
        '--source_seg_root',
        type=str,
        default="/home/haowei/Documents/vggt/training/dataset_aerialmd/cropped_seg",
        help="源分割掩码根目录"
    )
    parser.add_argument(
        '--source_npz',
        type=str,
        default="train.npz",
        help="源 NPZ 文件名"
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default="/home/haowei/Documents/vggt/training/dataset_aerialmd_single",
        help="输出数据集根目录"
    )
    parser.add_argument(
        '--pair_index',
        type=int,
        default=0,
        help="要提取的 pair 索引"
    )
    parser.add_argument(
        '--create_config',
        action='store_true',
        help="是否创建配置文件"
    )
    
    args = parser.parse_args()
    
    # 创建数据集
    output_root = create_single_pair_dataset(
        source_root=args.source_root,
        source_seg_root=args.source_seg_root,
        source_npz=args.source_npz,
        output_root=args.output_root,
        pair_index=args.pair_index
    )
    
    # 创建配置文件
    if args.create_config:
        create_config_for_single_pair(output_root)


if __name__ == "__main__":
    main()
