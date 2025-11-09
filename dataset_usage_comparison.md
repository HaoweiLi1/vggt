# AerialMegaDepth 数据集使用方法对比分析

## 📊 总体对比

| 方面 | Aerial-MegaDepth (DUSt3R) | VGGT |
|------|---------------------------|------|
| **用途** | DUSt3R/MASt3R 训练 | VGGT 多任务训练 |
| **基类** | BaseStereoViewDataset | BaseDataset |
| **图像数量** | 固定 2 张（立体对） | 灵活（2+ 张） |
| **坐标系统** | cam2world (OpenGL) | world2cam (OpenCV) |
| **数据增强** | crop_resize | 更复杂的处理流程 |

---

## 🔍 详细对比

### 1️⃣ RGB 图像加载

#### Aerial-MegaDepth (DUSt3R)
```python
# 文件路径
img_path = osp.join(seq_path, img + '.jpg')

# 加载方法
image = imread_cv2(img_path)  # DUSt3R 自定义函数

# 后处理
image, depthmap, intrinsics = self._crop_resize_if_necessary(
    image, depthmap, intrinsics, resolution, rng, info=(seq_path, img)
)

# 返回格式
views.append(dict(
    img=image,
    ...
))
```

**特点：**
- 使用 DUSt3R 的 `imread_cv2` 工具函数
- 简单的 crop/resize 操作
- 直接返回处理后的图像

#### VGGT
```python
# 文件路径（注意额外的 .jpg 后缀）
img_path = osp.join(scene_path, img_name + '.jpg')

# 加载方法
image = read_image_cv2(img_path, rgb=True)  # VGGT 工具函数

# 后处理（通过基类）
(
    image,
    depth_map,
    extri_opencv,
    intri_opencv,
    world_coords_points,
    cam_coords_points,
    point_mask,
    _,
) = self.process_one_image(
    image,
    depth_map,
    extri_opencv,
    intri_opencv,
    original_size,
    target_image_shape,
    filepath=osp.join(img_scene, img_name),
)

# 返回格式
batch = {
    "images": images,  # 列表
    ...
}
```

**特点：**
- 使用 VGGT 的 `read_image_cv2` 工具函数
- 复杂的 `process_one_image` 处理流程
- 生成额外的 3D 点云数据
- 支持批量处理多张图像

**⚠️ 关键差异：文件命名**
- Aerial-MegaDepth: `img_name.jpg`
- VGGT: `img_name.jpg.jpg` (预处理时添加了额外的 .jpg)

---

### 2️⃣ Depth Map 处理

#### Aerial-MegaDepth (DUSt3R)
```python
# 加载
depthmap = imread_cv2(osp.join(seq_path, img + ".exr"))

# 分割掩码应用（天空移除）
seg_root = self.ROOT.replace('megadepth_aerial_processed', 
                             'megadepth_aerial_processed_segmentation')
seg_path = osp.join(seg_root, scene, img + '.png')
segmap = imread_cv2(seg_path)
segmap = segmap[:, :, 0]  # 提取单通道
depthmap[segmap == 2] = 0  # 移除天空

# 离群值清理
min_depth, max_depth = np.percentile(depthmap, [0, 98])
depthmap[depthmap > max_depth] = 0

# Crop/Resize（与图像同步）
image, depthmap, intrinsics = self._crop_resize_if_necessary(...)
```

**处理流程：**
1. 加载 EXR 深度图
2. 应用分割掩码（强制，硬编码路径）
3. 百分位数过滤（0-98%）
4. 与图像同步 crop/resize

#### VGGT
```python
# 加载
depth_path = osp.join(scene_path, img_name + '.exr')
depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

# 多通道处理
if len(depth_map.shape) > 2:
    depth_map = depth_map[:, :, 0]

# 分割掩码应用（可选）
if self.remove_sky and self.segmentation_root:
    seg_path = osp.join(self.segmentation_root, scene, img_name + '.png')
    if osp.exists(seg_path):
        segmap = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        depth_map[segmap == 2] = 0  # 移除天空

# 离群值清理（更严格）
depth_map[depth_map > self.max_depth] = 0.0  # 硬阈值

valid_depths = depth_map[depth_map > 0]
if len(valid_depths) > 100:
    # 上限过滤
    depth_threshold = np.percentile(valid_depths, self.depth_percentile)
    depth_map[depth_map > depth_threshold] = 0.0
    
    # 下限过滤（额外步骤）
    min_threshold = np.percentile(valid_depths, 2)
    depth_map[depth_map < min_threshold] = 0.0

# 后续处理（通过 process_one_image）
```

**处理流程：**
1. 加载 EXR 深度图（使用 OpenCV）
2. 多通道检查
3. **可选**应用分割掩码（可配置）
4. 硬阈值过滤（max_depth）
5. 百分位数过滤（2%-98%，可配置）
6. 通过 `process_one_image` 进一步处理

**✅ 关键差异：**

| 特性 | Aerial-MegaDepth | VGGT |
|------|------------------|------|
| 分割掩码 | **强制使用**，硬编码路径 | **可选**，可配置路径 |
| 加载方式 | DUSt3R imread_cv2 | OpenCV cv2.imread |
| 硬阈值 | 无 | max_depth (1500-2000) |
| 百分位数 | [0, 98] 固定 | [2, depth_percentile] 可配置 |
| 下限过滤 | 无 | 有（2% 分位数） |

---

### 3️⃣ Camera Parameters 处理

#### Aerial-MegaDepth (DUSt3R)
```python
# 加载
camera_params = np.load(osp.join(seq_path, img + ".npz"))

# 提取参数
intrinsics = np.float32(camera_params['intrinsics'])
camera_pose = np.float32(camera_params['cam2world'])

# Crop/Resize 时调整内参
image, depthmap, intrinsics = self._crop_resize_if_necessary(...)

# 返回格式（cam2world）
views.append(dict(
    camera_pose=camera_pose,        # 4x4 cam2world (OpenGL)
    camera_intrinsics=intrinsics,   # 3x3 内参矩阵
    ...
))
```

**坐标系统：**
- **cam2world** (OpenGL 风格)
- 相机到世界的变换矩阵
- 用于 DUSt3R 的点云重建

#### VGGT
```python
# 加载
npz_path = osp.join(scene_path, img_name + '.npz')
camera_params = np.load(npz_path)

# 提取参数
intrinsics = camera_params['intrinsics'].astype(np.float32)
cam2world = camera_params['cam2world'].astype(np.float32)

# 坐标系转换（关键！）
world2cam = np.linalg.inv(cam2world)
extri_opencv = world2cam[:3, :]  # 取 3x4 部分
intri_opencv = K

# 通过 process_one_image 调整
(
    image,
    depth_map,
    extri_opencv,      # 调整后的外参
    intri_opencv,      # 调整后的内参
    world_coords_points,
    cam_coords_points,
    point_mask,
    _,
) = self.process_one_image(...)

# 返回格式（world2cam）
batch = {
    "extrinsics": extrinsics,  # 3x4 world2cam (OpenCV)
    "intrinsics": intrinsics,  # 3x3 内参矩阵
    "cam_points": cam_points,  # 相机坐标系点云
    "world_points": world_points,  # 世界坐标系点云
    ...
}
```

**坐标系统：**
- **world2cam** (OpenCV 风格)
- 世界到相机的变换矩阵
- 用于 VGGT 的多任务学习

**🔴 关键差异：坐标系统**

| 方面 | Aerial-MegaDepth | VGGT |
|------|------------------|------|
| 外参格式 | cam2world (4x4) | world2cam (3x4) |
| 坐标系 | OpenGL | OpenCV |
| 转换 | 无 | `np.linalg.inv(cam2world)` |
| 用途 | DUSt3R 点云重建 | 多任务相机估计 |

---

### 4️⃣ Segmentation Mask 使用

#### Aerial-MegaDepth (DUSt3R)
```python
# 路径（硬编码替换）
seg_root = self.ROOT.replace('megadepth_aerial_processed', 
                             'megadepth_aerial_processed_segmentation')
seg_path = osp.join(seg_root, scene, img + '.png')

# 加载（强制）
segmap = imread_cv2(seg_path)

# 验证（RGB 三通道相同）
assert (segmap[:, :, 0] == segmap[:, :, 1]).all()
assert (segmap[:, :, 0] == segmap[:, :, 2]).all()

# 提取单通道
segmap = segmap[:, :, 0]

# 应用到深度图
depthmap[segmap == 2] = 0  # ADE20k: 2 = 天空
```

**特点：**
- ✅ **强制使用**，无条件加载
- ⚠️ 硬编码路径替换规则
- ✅ 验证 RGB 通道一致性
- ✅ 使用 ADE20k 标准（天空 = 2）

#### VGGT
```python
# 路径（可配置）
if self.remove_sky and self.segmentation_root:
    seg_path = osp.join(self.segmentation_root, scene, img_name + '.png')
    
    # 文件存在性检查
    if osp.exists(seg_path):
        # 加载（直接读取灰度图）
        segmap = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        
        # 应用到深度图
        depth_map[segmap == 2] = 0  # ADE20k: 2 = 天空
```

**特点：**
- ✅ **可选使用**，通过 `remove_sky` 控制
- ✅ 灵活的路径配置（`segmentation_root`）
- ✅ 文件存在性检查
- ✅ 直接读取灰度图（更高效）
- ✅ 使用 ADE20k 标准（天空 = 2）

**✅ 关键差异：**

| 特性 | Aerial-MegaDepth | VGGT |
|------|------------------|------|
| 使用方式 | 强制 | 可选（`remove_sky=True`） |
| 路径配置 | 硬编码替换 | 配置参数 `segmentation_root` |
| 文件检查 | 无（会崩溃） | 有（`osp.exists`） |
| 加载方式 | RGB 后提取 | 直接灰度图 |
| 验证 | 断言检查 | 无 |

---

## 📦 数据结构对比

### Aerial-MegaDepth (DUSt3R) 输出
```python
views = [
    {
        'img': np.ndarray,              # RGB 图像
        'depthmap': np.ndarray,         # 深度图
        'camera_pose': np.ndarray,      # 4x4 cam2world
        'camera_intrinsics': np.ndarray,# 3x3 内参
        'dataset': 'MegaDepth',
        'label': str,                   # 场景相对路径
        'instance': str,                # 图像名称
    },
    # 第二个视图...
]
```

### VGGT 输出
```python
batch = {
    'seq_name': str,                    # 序列名称
    'ids': np.ndarray,                  # 图像 ID 列表
    'frame_num': int,                   # 帧数
    'images': [np.ndarray, ...],        # RGB 图像列表
    'depths': [np.ndarray, ...],        # 深度图列表
    'extrinsics': [np.ndarray, ...],    # 3x4 world2cam 列表
    'intrinsics': [np.ndarray, ...],    # 3x3 内参列表
    'cam_points': [np.ndarray, ...],    # 相机坐标系点云
    'world_points': [np.ndarray, ...],  # 世界坐标系点云
    'point_masks': [np.ndarray, ...],   # 有效点掩码
    'original_sizes': [tuple, ...],     # 原始尺寸
    'tracks': None,                     # 轨迹（可选）
    'track_masks': None,                # 轨迹掩码（可选）
}
```

---

## 🎯 核心差异总结

### 1. 设计哲学
- **Aerial-MegaDepth**: 简单、直接，专注于 DUSt3R 立体重建
- **VGGT**: 灵活、可配置，支持多任务学习

### 2. 坐标系统 🔴 **最重要的差异**
```python
# Aerial-MegaDepth (DUSt3R)
camera_pose = cam2world  # OpenGL 风格

# VGGT
extrinsics = np.linalg.inv(cam2world)[:3, :]  # OpenCV 风格
```

### 3. 分割掩码
- **Aerial-MegaDepth**: 强制使用，硬编码路径
- **VGGT**: 可选使用，灵活配置

### 4. 深度清理
- **Aerial-MegaDepth**: 简单（98% 分位数）
- **VGGT**: 严格（硬阈值 + 2%-98% 分位数）

### 5. 数据增强
- **Aerial-MegaDepth**: 基础 crop/resize
- **VGGT**: 复杂的 `process_one_image` 流程

### 6. 输出格式
- **Aerial-MegaDepth**: 视图列表（2 个）
- **VGGT**: 批次字典（2+ 个，包含点云）

---

## ✅ 一致性检查

### ✅ 相同之处
1. **数据源**: 都使用相同的预处理数据（.jpg, .exr, .npz）
2. **分割标准**: 都使用 ADE20k（天空 = 2）
3. **深度过滤**: 都使用百分位数方法
4. **NPZ 格式**: 都从 NPZ 加载 pairs 和 metadata

### ⚠️ 需要注意的差异
1. **坐标系统**: DUSt3R 使用 cam2world，VGGT 使用 world2cam
2. **文件命名**: VGGT 的图像文件有额外的 .jpg 后缀
3. **分割掩码**: VGGT 是可选的，需要在配置中启用
4. **深度阈值**: VGGT 使用更严格的过滤策略

---

## 🔧 配置建议

### VGGT 配置（与 Aerial-MegaDepth 对齐）
```yaml
- _target_: data.datasets.megadepth_aerial.MegaDepthAerialDataset
  split: train
  ROOT: /path/to/megadepth_aerial_processed
  split_file: train.npz
  segmentation_root: /path/to/megadepth_aerial_processed_segmentation  # ✅ 启用
  remove_sky: true                # ✅ 启用天空移除
  max_depth: 2000.0               # 适合航拍
  depth_percentile: 98.0          # 与 Aerial-MegaDepth 一致
  use_pairs: true
```

---

## 📝 结论

**VGGT 的实现与 Aerial-MegaDepth 在核心逻辑上是一致的**，主要差异在于：

1. **坐标系统转换**（OpenGL → OpenCV）- 这是架构需求
2. **更灵活的配置**（可选分割掩码、可调参数）
3. **更严格的数据清理**（额外的阈值过滤）
4. **更丰富的输出**（点云、掩码等）

这些差异是为了适应 VGGT 的多任务学习架构，而不是实现错误。只要正确配置 `segmentation_root` 和 `remove_sky=True`，VGGT 就能正确使用分割掩码来屏蔽天空区域。

**✅ 验证结果：VGGT 的实现是正确的！**
