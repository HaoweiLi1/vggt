# 关键修正总结 - 快速参考卡片

## 🚨 最重要的5个修正

### 1. VGGT外参方向 ❗❗❗

**错误认知**: VGGT输出c2w (camera-to-world)

**正确理解**: VGGT输出 **w2c (world-to-camera)**

**证据**: VGGT官方代码注释 "camera-from-world extrinsics"

**正确实现**:
```python
# ✅ 正确: 直接存储w2c
extrinsic_w2c = vggt_output  # [R|t] is w2c
json_data["extrinsic_w2c"] = {
    "R": R_w2c.tolist(),  # 3x3, world-to-camera
    "t": t_w2c.tolist()   # 3x1, world-to-camera
}

# ✅ 加载到Skyfall-GS (需要转置R)
R = np.transpose(R_w2c)  # Skyfall-GS存储R^T
T = t_w2c

# ✅ 如需c2w，在加载时转换
c2w = np.linalg.inv(w2c)
# 或
R_c2w = R_w2c.T
t_c2w = -R_w2c.T @ t_w2c
```

**验证方法**:
```python
# 相机中心应该在场景外围
C = -R_w2c.T @ t_w2c  # 如果是w2c
assert np.linalg.norm(C) > 10  # 不应该在原点
```

---

### 2. FoV单位 ❗❗

**错误认知**: FoV可能是角度

**正确理解**: FoV单位是 **弧度 (radian)**

**正确实现**:
```python
# ✅ 正确: FoV是弧度
fov_w, fov_h = vggt_fov  # in radians

# 转换为内参矩阵
fx = W / (2 * np.tan(fov_w / 2))
fy = H / (2 * np.tan(fov_h / 2))
cx = W / 2
cy = H / 2

# ❌ 错误: 不要这样做
# fx = W / (2 * np.tan(np.radians(fov_w) / 2))  # 错误!
```

**验证方法**:
```python
# 弧度范围检查
assert 0 < fov_w < np.pi, "FoV not in radian range"

# 转换为角度检查合理性
fov_deg = np.degrees(fov_w)
assert 10 < fov_deg < 120, "FoV out of reasonable range"
```

---

### 3. 深度置信度加权 ❗

**遗漏**: 原文档未强调置信度加权的必要性

**正确理解**: **必须**使用depth_conf作为像素权重

**正确实现**:
```python
# ✅ 正确: 置信度加权
depth_conf = vggt_depth_conf  # (H, W)

# 归一化置信度
conf_norm = (depth_conf - depth_conf.min()) / \
            (depth_conf.max() - depth_conf.min() + 1e-8)

# 创建mask（只保留高置信度）
conf_mask = conf_norm > 0.5  # 阈值可调
valid_mask = (gt_depth > 0) & conf_mask

# 提取权重
weights = conf_norm[valid_mask]

# 加权损失
if loss_type == "pearson":
    pred_mean = (pred_valid * weights).sum() / weights.sum()
    gt_mean = (gt_valid * weights).sum() / weights.sum()
    # ... 加权Pearson计算
```

**为什么重要**:
- 抑制低置信度区域（边缘、遮挡）
- 稳定训练收敛
- 提升深度监督效果

---

### 4. 跨视一致性的遮挡处理 ❗

**遗漏**: 原文档未处理遮挡点

**正确理解**: **必须**使用vis_scores过滤遮挡，使用鲁棒损失

**正确实现**:
```python
# ✅ 正确: 遮挡处理
tracks, vis_scores = extract_vggt_tracks(...)

for track, vis in zip(tracks, vis_scores):
    # 1. 过滤遮挡点
    visible_mask = vis > 0.5  # 可见性阈值
    if visible_mask.sum() < 2:
        continue  # 跳过遮挡严重的轨迹
    
    # 2. 只采样可见点的颜色
    visible_track = track[visible_mask]
    colors = [sample_bilinear(view, x, y) 
              for view, (x, y) in zip(views, visible_track)]
    colors = torch.stack(colors)
    
    # 3. 使用Huber loss（鲁棒）
    mean_color = colors.mean(dim=0)
    color_diff = colors - mean_color
    
    huber_loss = torch.where(
        color_diff.abs() < 0.1,
        0.5 * color_diff**2,
        0.1 * (color_diff.abs() - 0.05)
    )
    
    loss += huber_loss.mean()
```

**为什么重要**:
- 避免遮挡点干扰一致性
- 抑制离群点（误匹配）
- 提升跨视一致性质量

---

### 5. 场景归一化 ⚠️

**错误认知**: 硬编码target_radius=256

**正确理解**: 基于场景实际尺度**自适应**缩放

**正确实现**:
```python
# ✅ 正确: 自适应归一化
def normalize_scene(points_3d, method="percentile"):
    # 计算场景尺度
    if method == "percentile":
        center = np.median(points_3d, axis=0)
        distances = np.linalg.norm(points_3d - center, axis=1)
        radius = np.percentile(distances, 99)  # 99%分位数
    else:
        bbox_min = points_3d.min(axis=0)
        bbox_max = points_3d.max(axis=0)
        center = (bbox_min + bbox_max) / 2
        radius = np.linalg.norm(bbox_max - bbox_min) / 2
    
    # 自适应缩放（可选）
    target_radius = 256.0  # 或根据需求调整
    scale = target_radius / radius if radius > 0 else 1.0
    translate = -center
    
    # 记录到metadata（重要！）
    metadata = {
        "normalization": {
            "enabled": True,
            "scale": float(scale),
            "translate": translate.tolist(),
            "original_radius": float(radius)
        }
    }
    
    return normalized_points, scale, translate, metadata
```

**为什么重要**:
- 不同场景尺度差异大
- 硬编码可能导致数值不稳定
- 记录变换参数以便可逆

---

## 📋 坐标系约定总结

| 项目 | 坐标系 | 说明 |
|------|--------|------|
| **VGGT输出** | OpenCV | x右, y下, z前 |
| **外参方向** | w2c | world-to-camera |
| **FoV单位** | 弧度 | 不是角度 |
| **主点位置** | 图像中心 | (W/2, H/2) |
| **第一帧** | 世界参考系 | 用于点图/轨迹定义 |

| 项目 | Skyfall-GS | 说明 |
|------|-----------|------|
| **存储格式** | R^T, T | R是转置的w2c |
| **渲染器** | 可能OpenGL | 需要验证y/z翻转 |

---

## ✅ 快速验证清单

### 导出阶段
- [ ] JSON中标注pose_type="w2c_opencv"
- [ ] JSON中标注fov_unit="radian"
- [ ] 记录normalization参数到metadata
- [ ] 保存depth和depth_conf

### 加载阶段
- [ ] 验证pose_type是"w2c_opencv"
- [ ] R = np.transpose(R_w2c)
- [ ] 可视化相机位姿（8角箱体投影）
- [ ] 检查FoV转K矩阵的单位

### 训练阶段
- [ ] 深度损失使用depth_conf加权
- [ ] 创建有效mask（深度>0且有限）
- [ ] 跨视一致性使用vis_scores过滤
- [ ] 使用Huber loss抑制离群点

### 验收阶段
- [ ] 相机中心在场景外围（距离>10）
- [ ] FoV角度在合理范围（20-90度）
- [ ] 深度损失正常收敛
- [ ] 轨迹颜色方差<0.01

---

## 🔗 参考链接

- **VGGT论文**: Wang et al., CVPR 2025
- **VGGT代码**: https://github.com/facebookresearch/vggt
  - 关键文件: `vggt/utils/pose_enc.py` (外参解码)
  - 关键文件: `demo_colmap.py` (导出示例)
- **Skyfall-GS论文**: Lee et al., 2025
  - 关键章节: Sec 3.1 (重建阶段), Sec 3.2 (IDU)
  - 关键附录: Appendix A.1 (训练细节)

---

## 📞 遇到问题？

### 问题1: 相机位姿异常
→ 检查外参方向（w2c vs c2w）
→ 用可视化验证相机朝向

### 问题2: 深度损失不收敛
→ 检查置信度加权
→ 检查有效mask

### 问题3: 跨视一致性差
→ 检查vis_scores过滤
→ 检查Huber loss

### 问题4: FoV转换错误
→ 确认单位是弧度
→ 检查tan函数输入

---

**版本**: 2.0 (修正版)  
**日期**: 2025-11-06  
**状态**: ✅ 已审计并修正
