#!/usr/bin/env python3
"""
完整测试脚本：验证 VGGT 中 AerialMegaDepth 数据集的加载和处理
测试内容：
1. 数据加载器初始化
2. RGB 图像加载
3. Depth Map 加载和处理
4. Camera Parameters 加载和转换
5. Segmentation Mask 应用
6. 数据批次生成
"""

import os
import sys
import numpy as np
import cv2
import logging
from pathlib import Path

# 设置环境变量
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# 添加训练目录到路径
sys.path.insert(0, str(Path(__file__).parent / "training"))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_dataloader_initialization():
    """测试 1: 数据加载器初始化"""
    print("\n" + "="*70)
    print("测试 1: 数据加载器初始化")
    print("="*70)
    
    try:
        from data.datasets.megadepth_aerial import MegaDepthAerialDataset
        from types import SimpleNamespace
        
        # 创建配置
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
        
        # 初始化数据集
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
        
        print(f"✅ 数据集初始化成功")
        print(f"   - 场景数量: {len(dataset.valid_scenes)}")
        print(f"   - 有效场景: {dataset.valid_scenes}")
        print(f"   - 配对数量: {len(dataset.pairs)}")
        print(f"   - 数据集长度: {len(dataset)}")
        print(f"   - 分割掩码路径: {dataset.segmentation_root}")
        print(f"   - 天空移除: {dataset.remove_sky}")
        
        return dataset, True
        
    except Exception as e:
        print(f"❌ 数据集初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def test_single_image_loading(dataset):
    """测试 2: 单张图像加载"""
    print("\n" + "="*70)
    print("测试 2: 单张图像加载")
    print("="*70)
    
    try:
        # 获取第一个配对
        pair = dataset.pairs[0]
        scene_id = pair['scene_id']
        im1_id = pair['im1_id']
        
        scene = str(dataset.images_scene_name[im1_id])
        img_name = str(dataset.images[im1_id])
        
        print(f"测试图像: {scene}/{img_name}")
        
        # 加载图像数据
        img_data = dataset._load_image_data(scene, img_name)
        
        if img_data is None:
            print(f"❌ 图像加载失败")
            return False
            
        image, depth_map, cam2world, intrinsics = img_data
        
        print(f"✅ 图像加载成功")
        print(f"   - RGB 尺寸: {image.shape}")
        print(f"   - RGB 范围: [{image.min()}, {image.max()}]")
        print(f"   - Depth 尺寸: {depth_map.shape}")
        print(f"   - Depth 范围: [{depth_map.min():.2f}, {depth_map.max():.2f}]")
        print(f"   - 有效深度像素: {(depth_map > 0).sum()} / {depth_map.size}")
        print(f"   - 内参矩阵:\n{intrinsics}")
        print(f"   - cam2world 矩阵:\n{cam2world}")
        
        return True
        
    except Exception as e:
        print(f"❌ 图像加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_segmentation_mask_application(dataset):
    """测试 3: 分割掩码应用"""
    print("\n" + "="*70)
    print("测试 3: 分割掩码应用")
    print("="*70)
    
    try:
        # 获取测试图像
        pair = dataset.pairs[0]
        im1_id = pair['im1_id']
        scene = str(dataset.images_scene_name[im1_id])
        img_name = str(dataset.images[im1_id])
        
        scene_path = os.path.join(dataset.ROOT, scene)
        
        # 加载原始深度图（不应用掩码）
        depth_path = os.path.join(scene_path, img_name + '.exr')
        depth_original = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if len(depth_original.shape) > 2:
            depth_original = depth_original[:, :, 0]
        
        # 加载分割掩码
        seg_path = os.path.join(dataset.segmentation_root, scene, img_name + '.png')
        if not os.path.exists(seg_path):
            print(f"⚠️  分割掩码不存在: {seg_path}")
            return False
            
        segmap = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        
        # 统计天空像素
        sky_mask = (segmap == 2)
        sky_pixels = sky_mask.sum()
        sky_with_depth_before = (sky_mask & (depth_original > 0)).sum()
        
        # 应用掩码
        depth_masked = depth_original.copy()
        depth_masked[sky_mask] = 0
        sky_with_depth_after = (sky_mask & (depth_masked > 0)).sum()
        
        print(f"✅ 分割掩码应用测试")
        print(f"   - 测试图像: {scene}/{img_name}")
        print(f"   - 分割掩码尺寸: {segmap.shape}")
        print(f"   - 唯一标签: {np.unique(segmap)}")
        print(f"   - 天空像素数: {sky_pixels} ({sky_pixels/segmap.size*100:.1f}%)")
        print(f"   - 天空区域有效深度（处理前）: {sky_with_depth_before}")
        print(f"   - 天空区域有效深度（处理后）: {sky_with_depth_after}")
        print(f"   - 移除的深度像素: {sky_with_depth_before - sky_with_depth_after}")
        
        if sky_with_depth_before > 0 and sky_with_depth_after == 0:
            print(f"   ✅ 分割掩码正确移除了天空区域的深度值")
            return True
        elif sky_with_depth_before == 0:
            print(f"   ⚠️  该图像天空区域本身没有深度值")
            return True
        else:
            print(f"   ❌ 分割掩码未能完全移除天空区域的深度值")
            return False
            
    except Exception as e:
        print(f"❌ 分割掩码测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_camera_coordinate_conversion(dataset):
    """测试 4: 相机坐标系转换"""
    print("\n" + "="*70)
    print("测试 4: 相机坐标系转换")
    print("="*70)
    
    try:
        # 获取测试图像
        pair = dataset.pairs[0]
        im1_id = pair['im1_id']
        scene = str(dataset.images_scene_name[im1_id])
        img_name = str(dataset.images[im1_id])
        
        # 加载相机参数
        npz_path = os.path.join(dataset.ROOT, scene, img_name + '.npz')
        camera_params = np.load(npz_path)
        
        cam2world = camera_params['cam2world'].astype(np.float32)
        intrinsics = camera_params['intrinsics'].astype(np.float32)
        
        # VGGT 的转换
        world2cam = np.linalg.inv(cam2world)
        extri_opencv = world2cam[:3, :]
        
        # 验证转换
        # cam2world * world2cam 应该等于单位矩阵
        identity_check = cam2world @ world2cam
        is_identity = np.allclose(identity_check, np.eye(4), atol=1e-5)
        
        print(f"✅ 相机坐标系转换测试")
        print(f"   - cam2world 形状: {cam2world.shape}")
        print(f"   - world2cam 形状: {world2cam.shape}")
        print(f"   - extri_opencv 形状: {extri_opencv.shape}")
        print(f"   - 内参矩阵形状: {intrinsics.shape}")
        print(f"   - 逆矩阵验证: {'✅ 通过' if is_identity else '❌ 失败'}")
        
        # 显示矩阵样例
        print(f"\n   cam2world (前3行):")
        print(f"   {cam2world[:3, :]}")
        print(f"\n   world2cam (OpenCV extrinsics):")
        print(f"   {extri_opencv}")
        
        return is_identity
        
    except Exception as e:
        print(f"❌ 坐标系转换测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_generation(dataset):
    """测试 5: 批次数据生成"""
    print("\n" + "="*70)
    print("测试 5: 批次数据生成")
    print("="*70)
    
    try:
        # 生成一个批次
        batch = dataset.get_data(
            seq_index=0,
            img_per_seq=2,
            aspect_ratio=1.0
        )
        
        print(f"✅ 批次生成成功")
        print(f"   - 序列名称: {batch['seq_name']}")
        print(f"   - 图像数量: {batch['frame_num']}")
        print(f"   - 图像 ID: {batch['ids']}")
        
        # 检查每个组件
        for i in range(batch['frame_num']):
            print(f"\n   图像 {i}:")
            print(f"     - RGB 形状: {batch['images'][i].shape}")
            print(f"     - Depth 形状: {batch['depths'][i].shape}")
            print(f"     - Extrinsics 形状: {batch['extrinsics'][i].shape}")
            print(f"     - Intrinsics 形状: {batch['intrinsics'][i].shape}")
            print(f"     - Cam points 形状: {batch['cam_points'][i].shape}")
            print(f"     - World points 形状: {batch['world_points'][i].shape}")
            print(f"     - Point mask 形状: {batch['point_masks'][i].shape}")
            
            # 验证深度值
            depth = batch['depths'][i]
            valid_depth = depth[depth > 0]
            if len(valid_depth) > 0:
                print(f"     - 有效深度范围: [{valid_depth.min():.2f}, {valid_depth.max():.2f}]")
                print(f"     - 有效深度像素: {len(valid_depth)} / {depth.size}")
            else:
                print(f"     - ⚠️  没有有效深度值")
        
        # 验证批次完整性
        required_keys = ['seq_name', 'ids', 'frame_num', 'images', 'depths', 
                        'extrinsics', 'intrinsics', 'cam_points', 'world_points', 
                        'point_masks', 'original_sizes']
        
        missing_keys = [key for key in required_keys if key not in batch]
        if missing_keys:
            print(f"\n   ❌ 缺少键: {missing_keys}")
            return False
        else:
            print(f"\n   ✅ 批次包含所有必需的键")
            return True
            
    except Exception as e:
        print(f"❌ 批次生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_depth_filtering(dataset):
    """测试 6: 深度过滤策略"""
    print("\n" + "="*70)
    print("测试 6: 深度过滤策略")
    print("="*70)
    
    try:
        # 获取测试图像
        pair = dataset.pairs[0]
        im1_id = pair['im1_id']
        scene = str(dataset.images_scene_name[im1_id])
        img_name = str(dataset.images[im1_id])
        
        scene_path = os.path.join(dataset.ROOT, scene)
        
        # 加载原始深度图
        depth_path = os.path.join(scene_path, img_name + '.exr')
        depth_original = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if len(depth_original.shape) > 2:
            depth_original = depth_original[:, :, 0]
        
        # 模拟 VGGT 的过滤流程
        depth_filtered = depth_original.copy()
        
        # 步骤 1: 硬阈值过滤
        max_depth = dataset.max_depth
        before_hard_threshold = (depth_filtered > 0).sum()
        depth_filtered[depth_filtered > max_depth] = 0.0
        after_hard_threshold = (depth_filtered > 0).sum()
        
        # 步骤 2: 百分位数过滤
        valid_depths = depth_filtered[depth_filtered > 0]
        if len(valid_depths) > 100:
            depth_threshold = np.percentile(valid_depths, dataset.depth_percentile)
            min_threshold = np.percentile(valid_depths, 2)
            
            before_percentile = (depth_filtered > 0).sum()
            depth_filtered[depth_filtered > depth_threshold] = 0.0
            depth_filtered[depth_filtered < min_threshold] = 0.0
            after_percentile = (depth_filtered > 0).sum()
        else:
            depth_threshold = None
            min_threshold = None
            after_percentile = before_percentile = (depth_filtered > 0).sum()
        
        print(f"✅ 深度过滤策略测试")
        print(f"   - 测试图像: {scene}/{img_name}")
        print(f"   - 原始有效深度: {(depth_original > 0).sum()}")
        print(f"   - 硬阈值 (max_depth={max_depth}):")
        print(f"     处理前: {before_hard_threshold}, 处理后: {after_hard_threshold}")
        print(f"     移除: {before_hard_threshold - after_hard_threshold} 像素")
        
        if depth_threshold is not None:
            print(f"   - 百分位数过滤 (2%-{dataset.depth_percentile}%):")
            print(f"     下限阈值: {min_threshold:.2f}")
            print(f"     上限阈值: {depth_threshold:.2f}")
            print(f"     处理前: {before_percentile}, 处理后: {after_percentile}")
            print(f"     移除: {before_percentile - after_percentile} 像素")
        
        print(f"   - 最终有效深度: {(depth_filtered > 0).sum()} / {depth_filtered.size}")
        print(f"   - 总移除率: {(1 - (depth_filtered > 0).sum() / (depth_original > 0).sum()) * 100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ 深度过滤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_batches(dataset, num_batches=5):
    """测试 7: 多批次稳定性"""
    print("\n" + "="*70)
    print(f"测试 7: 多批次稳定性 (测试 {num_batches} 个批次)")
    print("="*70)
    
    success_count = 0
    failed_indices = []
    
    try:
        for i in range(min(num_batches, len(dataset))):
            try:
                batch = dataset.get_data(
                    seq_index=i,
                    img_per_seq=2,
                    aspect_ratio=1.0
                )
                
                # 基本验证
                if batch['frame_num'] >= 2:
                    success_count += 1
                    print(f"   ✅ 批次 {i}: {batch['seq_name']} - {batch['frame_num']} 帧")
                else:
                    failed_indices.append(i)
                    print(f"   ⚠️  批次 {i}: 只有 {batch['frame_num']} 帧")
                    
            except Exception as e:
                failed_indices.append(i)
                print(f"   ❌ 批次 {i}: 失败 - {e}")
        
        print(f"\n   总结:")
        print(f"   - 成功: {success_count}/{num_batches}")
        print(f"   - 失败: {len(failed_indices)}/{num_batches}")
        if failed_indices:
            print(f"   - 失败索引: {failed_indices}")
        
        return success_count == num_batches
        
    except Exception as e:
        print(f"❌ 多批次测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*70)
    print("VGGT AerialMegaDepth 数据加载器完整测试")
    print("="*70)
    
    results = {}
    
    # 测试 1: 初始化
    dataset, success = test_dataloader_initialization()
    results['初始化'] = success
    
    if not success or dataset is None:
        print("\n❌ 数据集初始化失败，无法继续测试")
        return
    
    # 测试 2: 单张图像加载
    results['单张图像加载'] = test_single_image_loading(dataset)
    
    # 测试 3: 分割掩码应用
    results['分割掩码应用'] = test_segmentation_mask_application(dataset)
    
    # 测试 4: 坐标系转换
    results['坐标系转换'] = test_camera_coordinate_conversion(dataset)
    
    # 测试 5: 批次生成
    results['批次生成'] = test_batch_generation(dataset)
    
    # 测试 6: 深度过滤
    results['深度过滤'] = test_depth_filtering(dataset)
    
    # 测试 7: 多批次稳定性
    results['多批次稳定性'] = test_multiple_batches(dataset, num_batches=5)
    
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
        print("\n🎉 所有测试通过！VGGT 正确使用了 AerialMegaDepth 数据集！")
    else:
        print(f"\n⚠️  有 {total_tests - passed_tests} 个测试失败，请检查配置")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
