# 语义分割掩码使用检查报告

## ✅ 检查结果：正确实现

VGGT 项目中的 AerialMegaDepth 数据加载器**已正确实现**语义分割掩码来屏蔽天空区域。

---

## 📋 实现细节对比

### 原始 Aerial-MegaDepth 实现
位置：`aerial-megadepth/data_generation/misc/megadepth_aerial.py`

```python
# 加载分割掩码
seg_root = self.ROOT.replace('megadepth_aerial_processed', 
                             'megadepth_aerial_processed_segmentation')
seg_path = osp.join(seg_root, scene, img + '.png')
segmap = imread_cv2(seg_path)

# 提取单通道
segmap = segmap[:, :, 0]

# 移除天空区域（ADE20k 标签 2 = 天空）
depthmap[segmap == 2] = 0

# 额外的离群值清理
min_depth, max_depth = np.percentile(depthmap, [0, 98])
depthmap[depthmap > max_depth] = 0
```

### VGGT 实现
位置：`vggt/training/data/datasets/megadepth_aerial.py`

```python
# 可选：加载并应用分割掩码来移除天空
if self.remove_sky and self.segmentation_root:
    seg_path = osp.join(self.segmentation_root, scene, img_name + '.png')
    if osp.exists(seg_path):
        segmap = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        # 移除天空（ADE20k 标签 2 = 天空）
        depth_map[segmap == 2] = 0

# 清理深度图
depth_map[depth_map > self.max_depth] = 0.0

# 使用百分位数移除离群值
valid_depths = depth_map[depth_map > 0]
if len(valid_depths) > 100:
    depth_threshold = np.percentile(valid_depths, self.depth_percentile)
    depth_map[depth_map > depth_threshold] = 0.0
    
    # 同时移除过近的深度（可能是噪声）
    min_threshold = np.percentile(valid_depths, 2)
    depth_map[depth_map < min_threshold] = 0.0
```

---

## 🔧 配置检查

### 1. 数据加载器参数
```python
class MegaDepthAerialDataset(BaseDataset):
    def __init__(
        self,
        ...
        segmentation_root: str = None,  # ✅ 支持分割掩码路径
        remove_sky: bool = True,        # ✅ 默认启用天空移除
        ...
    ):
```

### 2. 配置文件设置
位置：`vggt/training/config/default.yaml`

```yaml
# 训练集
- _target_: data.datasets.megadepth_aerial.MegaDepthAerialDataset
  split: train
  ROOT: /home/haowei/Documents/vggt/training/dataset_aerialmd/cropped
  split_file: train.npz
  segmentation_root: /home/haowei/Documents/vggt/training/dataset_aerialmd/cropped_seg  # ✅ 已配置
  max_depth: 2000.0
  depth_percentile: 98.0

# 验证集
- _target_: data.datasets.megadepth_aerial.MegaDepthAerialDataset
  split: val
  ROOT: /home/haowei/Documents/vggt/training/dataset_aerialmd/cropped
  split_file: val.npz
  segmentation_root: /home/haowei/Documents/vggt/training/dataset_aerialmd/cropped_seg  # ✅ 已配置
  max_depth: 2000.0
  depth_percentile: 98.0
```

### 3. 数据文件验证
```bash
✅ 分割掩码目录存在：
   /home/haowei/Documents/vggt/training/dataset_aerialmd/cropped_seg/

✅ 场景目录：0000, 0001, 0002, 0003, 0015

✅ 掩码格式：PNG 灰度图像 (518x518)

✅ 文件命名：与原始图像对应 (例如：0001_001.jpeg.png)
```

---

## 🧪 功能测试结果

测试场景：`0001/0001_001.jpeg`

```
深度图尺寸：518 x 518
深度范围：[0.00, 444.04]
有效深度像素（处理前）：183,189 / 268,324

分割掩码尺寸：518 x 518
语义标签：[0, 1, 2, 12, 17, 43, 132]
天空像素（标签=2）：77,586 / 268,324 (28.9%)

天空区域有效深度（处理前）：5,512 像素
天空区域有效深度（处理后）：0 像素

移除的像素数：5,512
```

**结论：✅ 分割掩码成功移除了所有天空区域的深度值**

---

## 🎯 关键差异与改进

### VGGT 相比原始实现的改进：

1. **条件检查更完善**
   - 检查 `remove_sky` 标志
   - 检查 `segmentation_root` 是否设置
   - 检查文件是否存在

2. **直接读取灰度图**
   ```python
   # VGGT: 直接读取灰度图，更高效
   segmap = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
   
   # 原始: 读取 RGB 后提取单通道
   segmap = imread_cv2(seg_path)
   segmap = segmap[:, :, 0]
   ```

3. **额外的深度清理**
   - 移除过近的深度（< 2% 分位数）
   - 更灵活的 `max_depth` 和 `depth_percentile` 参数

4. **配置灵活性**
   - 可以通过配置文件轻松启用/禁用
   - 支持不同的分割掩码路径

---

## ✅ 最终结论

**VGGT 项目已正确实现语义分割掩码功能，用于在训练时屏蔽天空区域的深度值。**

实现方式：
- ✅ 使用 ADE20k 标准（天空 = 标签 2）
- ✅ 在数据加载时应用掩码
- ✅ 配置文件中正确设置路径
- ✅ 数据文件已准备就绪
- ✅ 功能测试通过

该实现与原始 Aerial-MegaDepth 论文的方法一致，并在某些方面有所改进。
