# AerialMegaDepth 数据集使用对比 - 快速总结

## ✅ 核心结论

**VGGT 的实现与 Aerial-MegaDepth 在数据使用方法上基本一致，主要差异是为了适应不同的训练架构。**

---

## 🔑 关键对比表

| 组件 | Aerial-MegaDepth (DUSt3R) | VGGT | 一致性 |
|------|---------------------------|------|--------|
| **RGB 加载** | `imread_cv2(img + '.jpg')` | `read_image_cv2(img + '.jpg.jpg')` | ⚠️ 文件名差异 |
| **Depth 加载** | `imread_cv2(img + '.exr')` | `cv2.imread(img + '.exr')` | ✅ 格式一致 |
| **Camera Params** | `cam2world` (OpenGL) | `world2cam` (OpenCV) | 🔴 坐标系不同 |
| **Segmentation** | 强制使用，硬编码路径 | 可选使用，可配置路径 | ⚠️ 灵活性不同 |
| **天空移除** | `depthmap[segmap == 2] = 0` | `depth_map[segmap == 2] = 0` | ✅ 完全一致 |
| **深度过滤** | 98% 分位数 | 2%-98% 分位数 + 硬阈值 | ⚠️ VGGT 更严格 |

---

## 📋 详细对比

### 1. RGB 图像
```python
# Aerial-MegaDepth
image = imread_cv2(osp.join(seq_path, img + '.jpg'))

# VGGT
image = read_image_cv2(osp.join(scene_path, img_name + '.jpg'))
# 注意：VGGT 预处理时添加了额外的 .jpg 后缀
```
**差异**: 文件命名约定不同  
**影响**: 无，只要数据预处理正确

---

### 2. Depth Map
```python
# Aerial-MegaDepth
depthmap = imread_cv2(osp.join(seq_path, img + ".exr"))
min_depth, max_depth = np.percentile(depthmap, [0, 98])
depthmap[depthmap > max_depth] = 0

# VGGT
depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
depth_map[depth_map > self.max_depth] = 0.0  # 硬阈值
depth_threshold = np.percentile(valid_depths, self.depth_percentile)
depth_map[depth_map > depth_threshold] = 0.0
min_threshold = np.percentile(valid_depths, 2)
depth_map[depth_map < min_threshold] = 0.0  # 额外的下限过滤
```
**差异**: VGGT 有额外的硬阈值和下限过滤  
**影响**: VGGT 的深度清理更严格，可能更适合航拍数据

---

### 3. Camera Parameters 🔴 **最重要的差异**
```python
# Aerial-MegaDepth (DUSt3R)
camera_params = np.load(osp.join(seq_path, img + ".npz"))
intrinsics = np.float32(camera_params['intrinsics'])
camera_pose = np.float32(camera_params['cam2world'])  # 4x4 cam2world

# VGGT
camera_params = np.load(npz_path)
intrinsics = camera_params['intrinsics'].astype(np.float32)
cam2world = camera_params['cam2world'].astype(np.float32)
world2cam = np.linalg.inv(cam2world)  # 转换坐标系
extri_opencv = world2cam[:3, :]  # 3x4 world2cam
```

**关键差异**:
- **Aerial-MegaDepth**: 使用 `cam2world` (OpenGL 风格)，用于 DUSt3R 点云重建
- **VGGT**: 转换为 `world2cam` (OpenCV 风格)，用于相机姿态估计

**为什么不同**:
- DUSt3R: 需要从相机坐标投影到世界坐标
- VGGT: 需要从世界坐标投影到相机坐标（标准 CV 流程）

**影响**: 这是架构需求，不是错误

---

### 4. Segmentation Mask ✅ **核心功能一致**
```python
# Aerial-MegaDepth (强制使用)
seg_root = self.ROOT.replace('megadepth_aerial_processed', 
                             'megadepth_aerial_processed_segmentation')
seg_path = osp.join(seg_root, scene, img + '.png')
segmap = imread_cv2(seg_path)
segmap = segmap[:, :, 0]  # 提取单通道
depthmap[segmap == 2] = 0  # ADE20k: 天空 = 2

# VGGT (可选使用)
if self.remove_sky and self.segmentation_root:
    seg_path = osp.join(self.segmentation_root, scene, img_name + '.png')
    if osp.exists(seg_path):
        segmap = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        depth_map[segmap == 2] = 0  # ADE20k: 天空 = 2
```

**核心逻辑**: ✅ **完全一致** - 都使用 ADE20k 标准（天空 = 2）

**差异**:
| 特性 | Aerial-MegaDepth | VGGT |
|------|------------------|------|
| 使用方式 | 强制 | 可选（需配置） |
| 路径 | 硬编码替换 | 配置参数 |
| 错误处理 | 无（会崩溃） | 有（文件检查） |
| 加载方式 | RGB → 单通道 | 直接灰度图 |

**优势**: VGGT 更灵活、更健壮

---

## 🎯 实际使用验证

### 测试结果（场景 0001/0001_001.jpeg）
```
✅ 深度图尺寸: 518 x 518
✅ 天空像素: 77,586 (28.9%)
✅ 天空区域有效深度（处理前）: 5,512 像素
✅ 天空区域有效深度（处理后）: 0 像素
✅ 成功移除: 5,512 像素

结论: 分割掩码正确工作！
```

---

## 📊 配置对比

### Aerial-MegaDepth 配置
```python
dataset = MegaDepthAerial(
    split='train', 
    ROOT="/mnt/slarge2/megadepth_aerial_processed", 
    split_file='aerial_megadepth_train_part1.npz',
    resolution=224, 
    aug_crop=16
)
# 分割掩码路径硬编码
```

### VGGT 配置
```yaml
- _target_: data.datasets.megadepth_aerial.MegaDepthAerialDataset
  split: train
  ROOT: /home/haowei/Documents/vggt/training/dataset_aerialmd/cropped
  split_file: train.npz
  segmentation_root: /home/haowei/Documents/vggt/training/dataset_aerialmd/cropped_seg  # ✅
  remove_sky: true          # ✅ 默认启用
  max_depth: 2000.0
  depth_percentile: 98.0
  use_pairs: true
```

---

## ✅ 一致性检查清单

| 检查项 | 状态 | 说明 |
|--------|------|------|
| RGB 加载 | ✅ | 格式一致，文件名约定不同 |
| Depth 加载 | ✅ | EXR 格式，VGGT 过滤更严格 |
| Camera Params | ⚠️ | 坐标系不同（架构需求） |
| Segmentation Mask | ✅ | 核心逻辑完全一致 |
| 天空移除 | ✅ | ADE20k 标准（天空 = 2） |
| 深度过滤 | ✅ | 都使用百分位数，VGGT 更严格 |
| NPZ 格式 | ✅ | 完全兼容 |
| 数据路径 | ✅ | 已正确配置 |

---

## 🚀 最终结论

### ✅ VGGT 的实现是正确的！

**核心功能一致**:
1. ✅ RGB、Depth、Camera Params 加载方式正确
2. ✅ Segmentation Mask 使用 ADE20k 标准
3. ✅ 天空移除逻辑完全一致（`segmap == 2`）
4. ✅ 深度过滤使用百分位数方法

**差异是合理的**:
1. 坐标系转换（OpenGL → OpenCV）- 架构需求
2. 可选分割掩码 - 更灵活
3. 更严格的深度过滤 - 更适合航拍数据
4. 更丰富的输出 - 支持多任务学习

**配置正确**:
- ✅ `segmentation_root` 已设置
- ✅ `remove_sky=True` 已启用
- ✅ 数据文件已准备就绪
- ✅ 功能测试通过

### 📝 建议

1. **保持当前配置** - 已经正确实现
2. **监控训练** - 确保天空区域不影响深度损失
3. **可选优化** - 如果需要，可以调整 `depth_percentile` 参数

---

## 📚 参考文档

- 详细对比: `dataset_usage_comparison.md`
- 分割掩码检查: `segmentation_check_report.md`
- 测试脚本: `test_segmentation_mask.py`
