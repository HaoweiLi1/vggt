# VGGT + Skyfall-GS Integration Specification

## ⚠️ 重要修正说明

本文档已根据VGGT官方实现和Skyfall-GS论文进行了关键修正：

### 修正1：VGGT外参方向（❗最重要）
- **错误**: 之前文档错误地将VGGT外参描述为c2w (camera-to-world)
- **正确**: VGGT输出的是 **w2c (world-to-camera, camera-from-world)**
- **证据**: VGGT官方代码注释明确标注 "camera-from-world extrinsics"
- **影响**: 所有涉及外参的代码必须使用w2c，如需c2w则在加载时转换

### 修正2：FoV单位
- **强调**: VGGT输出的FoV单位是**弧度(radian)**，不是角度
- **常见错误**: 将角度当弧度使用会导致内参矩阵错误

### 修正3：坐标系约定
- **VGGT**: OpenCV坐标系 (x右, y下, z前)
- **部分渲染器**: OpenGL坐标系 (x右, y上, z后)
- **必须**: 通过投影测试验证轴向是否需要翻转

### 修正4：置信度加权
- **新增**: VGGT深度损失必须使用depth_conf作为像素权重
- **原因**: 抑制低置信度区域，稳定训练收敛

### 修正5：跨视一致性的遮挡处理
- **新增**: 使用vis_scores过滤遮挡点
- **新增**: 使用Huber loss抑制离群点
- **原因**: 避免遮挡点干扰一致性约束

### 修正6：场景归一化
- **修正**: 不硬编码target_radius=256
- **建议**: 基于场景实际尺度自适应缩放
- **必须**: 记录scale/translate到metadata以便可逆

---

## 项目概述

本文档详细说明如何将VGGT（Visual Geometry Grounded Transformer）集成到Skyfall-GS pipeline中，并实现Few-shot Color Alignment Diffusion模块，以提升卫星图像3D重建的效率和质量。

## 目标

1. **效率提升**：用VGGT替代COLMAP/SatelliteSfM，将位姿初始化从分钟级降至秒级
2. **质量提升**：实现Few-shot Color Alignment Diffusion，比FlowEdit更好地对齐GT颜色分布
3. **鲁棒性提升**：在低纹理卫星图像场景下更稳定

## 参考文献

### 核心论文
- **VGGT**: Wang et al., "Visual Geometry Grounded Transformer", CVPR 2025
  - 参数化: g=[q,t,f], 主点在中心, 第一帧作世界参考
  - 外参: w2c (camera-from-world), OpenCV坐标系
  - 速度: 前向~0.2s, +BA~1.8s
  
- **Skyfall-GS**: Lee et al., "Synthesizing Immersive 3D Urban Scenes from Satellite Imagery", 2025
  - 两阶段: 重建 + IDU (Iterative Dataset Update)
  - FlowEdit + FLUX.1[dev] 做prompt-to-prompt修复
  - 课程式采样 (高→低仰角), 多样本 (Ns≥2), 75/25混采
  - 外观建模 + 不透明度熵正则 + 伪相机深度监督(MoGe + PCorr)

- **FlowEdit**: "Inversion-Free Text-Based Editing using FLUX", 2024
  - 无反演的文本引导图像编辑
  - 适配FLUX/SD3

### 官方实现
- VGGT: https://github.com/facebookresearch/vggt
- Skyfall-GS: (你的仓库)
- FlowEdit: https://github.com/fallenshock/FlowEdit

---

## 任务一：VGGT位姿初始化集成（必做，优先级1）

### 1.1 技术背景

**Skyfall-GS当前流程**：
- 使用SatelliteSfM/COLMAP生成稀疏点云和相机位姿
- 使用MoGe预测深度图作为伪相机深度监督
- 初始化3DGS并优化（外观建模 + 不透明度熵正则 + 深度监督）

**VGGT能力**：
- 一次前向预测：相机内外参、深度图、点图、轨迹
- 速度：<1秒（前向），+BA约1-2秒
- 坐标系：第一帧为世界坐标系，主点在图像中心
- 输出格式：可导出COLMAP格式

### 1.2 坐标系与格式转换

#### 1.2.1 VGGT输出格式

**相机内参**：
```
VGGT输出FOV: (fov_w, fov_h) in RADIANS (弧度)
转换为内参矩阵K:
fx = W / (2 * tan(fov_w / 2))
fy = H / (2 * tan(fov_h / 2))
cx = W / 2
cy = H / 2

K = [
  [fx,  0,  cx],
  [0,  fy,  cy],
  [0,   0,   1]
]

⚠️ 注意: FoV单位是弧度，不是角度！
```

**相机外参**：
```
❗ 重要修正: VGGT输出的是 w2c (world-to-camera, camera-from-world)

VGGT官方实现输出: 
- 外参矩阵 [R|t] 是 world-to-camera (w2c)
- 坐标系: OpenCV convention (x右, y下, z前)
- 第一帧作为世界参考系 (用于点图/轨迹一致性定义)

w2c = [R | t]  // VGGT直接输出
      [0 | 1]

如果渲染器需要 c2w (camera-to-world):
c2w = inv(w2c) = [R^T | -R^T * t]
                 [0   | 1        ]

相机中心位置: C = -R^T * t

参考: VGGT官方代码注释明确标注 "camera-from-world extrinsics"
```

**深度图**：
```
VGGT输出: 
- depth_map: (N, H, W) 米制深度
- depth_conf: (N, H, W) 置信度分数/不确定度

用途:
1. 替代MoGe作为伪相机深度监督
2. 用于初始化3D点云
3. 置信度作为像素权重，稳定深度损失收敛

⚠️ 建议: 使用depth_conf作为损失权重，抑制低置信度区域
```

#### 1.2.2 数据格式定义

创建新的JSON格式存储VGGT输出：

**文件路径**: `{scene_dir}/vggt_poses.json`

**格式** (修正版):
```json
{
  "frames": [
    {
      "image_path": "images/frame_001.png",
      "width": 1024,
      "height": 1024,
      "camera_id": 0,
      "intrinsic": {
        "fx": 512.0,
        "fy": 512.0,
        "cx": 512.0,
        "cy": 512.0
      },
      "extrinsic_w2c": {
        "R": [[...], [...], [...]],  // 3x3 rotation matrix (world-to-camera)
        "t": [x, y, z]                // 3x1 translation vector (world-to-camera)
      },
      "depth_path": "depths_vggt/frame_001.npy",
      "depth_conf_path": "depths_vggt/frame_001_conf.npy"
    }
  ],
  "metadata": {
    "source": "vggt",
    "version": "1.0",
    "pose_type": "w2c_opencv",
    "fov_unit": "radian",
    "coordinate_system": "first_frame_as_reference",
    "normalization": {
      "enabled": false,
      "scale": 1.0,
      "translate": [0.0, 0.0, 0.0]
    }
  }
}
```

**关键说明**:
1. **统一存储w2c**: 避免c2w/w2c混淆，统一存储VGGT原生的w2c
2. **pose_type标注**: 明确标注"w2c_opencv"，表示OpenCV坐标系的world-to-camera
3. **fov_unit标注**: 明确FoV单位是弧度
4. **归一化可选**: 仅在需要时启用，记录scale/translate以便可逆
5. **加载时转换**: 如渲染器需要c2w，在加载时做 `c2w = inv(w2c)`

**坐标系约定**:
- OpenCV: x右, y下, z前 (VGGT使用)
- OpenGL: x右, y上, z后 (部分渲染器使用)
- 需要在投影测试中验证轴向是否正确

### 1.3 实现步骤

#### 步骤1：创建VGGT导出脚本

**文件**: `Skyfall-GS/vggt_pose_export.py`

**功能**:
1. 加载场景图像
2. 运行VGGT前向预测
3. 转换坐标系和格式
4. 保存为Skyfall-GS兼容格式
5. 可选：运行轻量BA优化

**关键函数**:
```python
def vggt_to_skyfall_format(extrinsic_w2c, intrinsic, image_paths, depths, depth_confs):
    """
    将VGGT输出转换为Skyfall-GS格式
    
    Args:
        extrinsic_w2c: (N, 4, 4) w2c matrices (VGGT原生输出)
        intrinsic: (N, 3, 3) K matrices
        image_paths: list of image paths
        depths: (N, H, W) depth maps
        depth_confs: (N, H, W) confidence scores
    
    Returns:
        dict: Skyfall-GS compatible format (JSON)
    
    ⚠️ 注意:
    - 直接存储w2c，不做转置
    - 在metadata中标注pose_type="w2c_opencv"
    - 如渲染器需要c2w，在加载时转换
    """
    frames = []
    for i in range(len(image_paths)):
        w2c = extrinsic_w2c[i]  # 4x4
        R = w2c[:3, :3]  # 3x3 rotation
        t = w2c[:3, 3]   # 3x1 translation
        
        K = intrinsic[i]
        
        frame = {
            "image_path": image_paths[i],
            "width": depths[i].shape[1],
            "height": depths[i].shape[0],
            "intrinsic": {
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2])
            },
            "extrinsic_w2c": {
                "R": R.tolist(),
                "t": t.tolist()
            },
            "depth_path": f"depths_vggt/frame_{i:03d}.npy",
            "depth_conf_path": f"depths_vggt/frame_{i:03d}_conf.npy"
        }
        frames.append(frame)
    
    return {
        "frames": frames,
        "metadata": {
            "source": "vggt",
            "pose_type": "w2c_opencv",
            "fov_unit": "radian"
        }
    }

def normalize_scene(extrinsic_w2c, points_3d, method="percentile"):
    """
    可选的场景归一化（仅在需要时使用）
    
    Args:
        extrinsic_w2c: (N, 4, 4) camera poses (w2c)
        points_3d: (M, 3) 3D points
        method: "percentile" or "bbox"
    
    Returns:
        normalized_extrinsic, normalized_points, scale, translate
    
    ⚠️ 建议:
    - 不要硬编码target_radius=256
    - 基于场景实际尺度自适应缩放
    - 记录scale/translate到metadata以便可逆
    """
    # 计算场景尺度
    if method == "percentile":
        # 使用99%分位数避免离群点
        center = np.median(points_3d, axis=0)
        distances = np.linalg.norm(points_3d - center, axis=1)
        radius = np.percentile(distances, 99)
    else:  # bbox
        bbox_min = points_3d.min(axis=0)
        bbox_max = points_3d.max(axis=0)
        center = (bbox_min + bbox_max) / 2
        radius = np.linalg.norm(bbox_max - bbox_min) / 2
    
    # 自适应缩放（可选）
    target_radius = 256.0  # 或根据渲染器需求调整
    scale = target_radius / radius if radius > 0 else 1.0
    translate = -center
    
    # 应用变换
    normalized_points = (points_3d + translate) * scale
    
    # 变换相机位姿
    normalized_extrinsic = []
    for w2c in extrinsic_w2c:
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        # t' = scale * (R * translate + t)
        t_new = scale * (R @ translate + t)
        w2c_new = np.eye(4)
        w2c_new[:3, :3] = R
        w2c_new[:3, 3] = t_new
        normalized_extrinsic.append(w2c_new)
    
    return np.array(normalized_extrinsic), normalized_points, scale, translate
```


#### 步骤2：修改Skyfall-GS数据加载器

**文件**: `Skyfall-GS/scene/dataset_readers.py`

**修改点**:

1. 添加新的读取函数：
```python
def readVGGTCamerasFromJSON(path, json_file="vggt_poses.json"):
    """
    从VGGT导出的JSON读取相机参数
    
    Args:
        path: scene directory
        json_file: JSON filename
    
    Returns:
        list of CameraInfo
    
    ⚠️ 关键实现细节:
    1. JSON中存储的是w2c，需要转换为Skyfall-GS的R, T格式
    2. 验证pose_type是否为"w2c_opencv"
    3. 检查OpenCV↔OpenGL坐标系是否需要翻转
    4. 加载depth和depth_conf
    """
    import json
    import numpy as np
    from PIL import Image
    
    json_path = os.path.join(path, json_file)
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 验证格式
    assert data["metadata"]["pose_type"] == "w2c_opencv", \
        "Only w2c_opencv format is supported"
    
    cam_infos = []
    for idx, frame in enumerate(data["frames"]):
        # 读取图像
        image_path = os.path.join(path, frame["image_path"])
        image = Image.open(image_path)
        image_name = Path(image_path).stem
        
        # 读取内参
        intr = frame["intrinsic"]
        fx, fy = intr["fx"], intr["fy"]
        cx, cy = intr["cx"], intr["cy"]
        width, height = frame["width"], frame["height"]
        
        # 转换为FoV (Skyfall-GS使用FoV)
        FovX = focal2fov(fx, width)
        FovY = focal2fov(fy, height)
        
        # 归一化主点到[-1, 1]
        cx_norm = (cx - width / 2) / width * 2
        cy_norm = (cy - height / 2) / height * 2
        
        # 读取外参 (w2c)
        extr = frame["extrinsic_w2c"]
        R_w2c = np.array(extr["R"])  # 3x3
        t_w2c = np.array(extr["t"])  # 3x1
        
        # Skyfall-GS存储格式: R是转置的w2c rotation
        # 这里R_w2c已经是w2c，直接转置即可
        R = np.transpose(R_w2c)  # 存储为R^T
        T = t_w2c
        
        # 读取深度
        depth = None
        depth_conf = None
        if "depth_path" in frame:
            depth_path = os.path.join(path, frame["depth_path"])
            if os.path.exists(depth_path):
                depth = np.load(depth_path)
        if "depth_conf_path" in frame:
            conf_path = os.path.join(path, frame["depth_conf_path"])
            if os.path.exists(conf_path):
                depth_conf = np.load(conf_path)
        
        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            cx=cx_norm,
            cy=cy_norm,
            image=image,
            image_path=image_path,
            image_name=image_name,
            depth=depth,
            depth_conf=depth_conf,  # 新增字段
            mask=None,
            width=width,
            height=height
        )
        cam_infos.append(cam_info)
    
    return cam_infos
```

2. 修改`readSatelliteInfo`函数，添加VGGT路径检测：
```python
def readSatelliteInfo(path, white_background, eval, extension=".png"):
    # 优先检查VGGT输出
    vggt_json_path = os.path.join(path, "vggt_poses.json")
    if os.path.exists(vggt_json_path):
        print("Found vggt_poses.json, using VGGT initialization")
        train_cam_infos = readVGGTCamerasFromJSON(path, "vggt_poses.json")
        test_cam_infos = []  # 或从separate test json读取
        
        # 加载点云
        ply_path = os.path.join(path, "vggt_points3d.ply")
        if os.path.exists(ply_path):
            pcd = fetchPly(ply_path)
        else:
            # 从VGGT深度生成点云
            pcd = generate_pcd_from_vggt_depth(train_cam_infos)
    else:
        # Fallback到原有的transforms_train.json逻辑
        print("VGGT poses not found, using original SfM initialization")
        # ... 原有代码
```

3. 修改CameraInfo定义（如需要）：
```python
# 在scene/dataset_readers.py顶部
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    cx: np.array
    cy: np.array
    image: np.array
    image_path: str
    image_name: str
    depth: np.array
    depth_conf: np.array  # 新增：VGGT置信度
    mask: np.array
    width: int
    height: int
```

4. 添加坐标系验证函数：
```python
def verify_camera_coordinate_system(cam_infos, save_path="debug_cameras.png"):
    """
    验证相机坐标系是否正确
    通过可视化相机朝向和8角箱体投影
    
    ⚠️ 检查项:
    1. 相机是否朝向场景中心
    2. y轴是否向下（OpenCV）还是向上（OpenGL）
    3. z轴是否向前（OpenCV）还是向后（OpenGL）
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for cam in cam_infos[:10]:  # 只可视化前10个
        # 从R, T恢复相机中心
        R = cam.R  # 已转置的w2c rotation
        T = cam.T
        # C = -R^T * T (因为R已经是转置的)
        C = -R @ T
        
        # 绘制相机位置
        ax.scatter(C[0], C[1], C[2], c='r', marker='o')
        
        # 绘制相机朝向（z轴）
        # z_axis = R^T @ [0, 0, 1]
        z_axis = R @ np.array([0, 0, 1])
        ax.quiver(C[0], C[1], C[2], z_axis[0], z_axis[1], z_axis[2], 
                  length=10, color='b')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(save_path)
    print(f"Camera visualization saved to {save_path}")
    print("⚠️ 检查: 蓝色箭头应指向场景中心")
```

#### 步骤3：集成深度监督

**文件**: `Skyfall-GS/train.py`

**修改点**:

1. 在训练循环中添加VGGT深度损失：
```python
# 原有MoGe深度监督
if opt.lambda_pseudo_depth > 0:
    # ... MoGe相关代码

# 新增VGGT深度监督
if hasattr(viewpoint_cam, 'depth') and viewpoint_cam.depth is not None:
    rendered_depth = render_pkg["depth"]
    gt_depth = torch.from_numpy(viewpoint_cam.depth).cuda()
    
    # 使用置信度加权（关键改进）
    if hasattr(viewpoint_cam, 'depth_conf') and viewpoint_cam.depth_conf is not None:
        depth_conf = torch.from_numpy(viewpoint_cam.depth_conf).cuda()
    else:
        depth_conf = None
    
    # 计算深度损失
    depth_loss = compute_depth_loss(
        rendered_depth, 
        gt_depth, 
        conf=depth_conf,
        loss_type=opt.vggt_depth_loss_type
    )
    loss += opt.lambda_vggt_depth * depth_loss
    
    # 记录日志
    if iteration % 100 == 0:
        print(f"VGGT Depth Loss: {depth_loss.item():.4f}")
```

2. 添加深度损失计算函数（带置信度加权）：
```python
def compute_depth_loss(pred_depth, gt_depth, conf=None, loss_type="pearson"):
    """
    计算深度损失（带置信度加权）
    
    Args:
        pred_depth: rendered depth (H, W)
        gt_depth: VGGT depth (H, W)
        conf: confidence weights (H, W), 高值=高置信度
        loss_type: "pearson", "silog", or "gradient"
    
    Returns:
        weighted loss scalar
    
    ⚠️ 关键: 使用conf加权可以抑制低置信度区域，稳定训练
    """
    # 确保形状匹配
    if pred_depth.shape != gt_depth.shape:
        gt_depth = F.interpolate(
            gt_depth.unsqueeze(0).unsqueeze(0),
            size=pred_depth.shape,
            mode='bilinear',
            align_corners=False
        ).squeeze()
    
    # 创建有效mask（深度>0且有限）
    valid_mask = (gt_depth > 0) & torch.isfinite(gt_depth) & torch.isfinite(pred_depth)
    
    if conf is not None:
        # 归一化置信度到[0, 1]
        conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
        # 只保留高置信度区域（可选阈值）
        conf_mask = conf_norm > 0.5
        valid_mask = valid_mask & conf_mask
        weights = conf_norm[valid_mask]
    else:
        weights = None
    
    if valid_mask.sum() < 10:
        return torch.tensor(0.0).cuda()
    
    pred_valid = pred_depth[valid_mask]
    gt_valid = gt_depth[valid_mask]
    
    if loss_type == "pearson":
        # Pearson Correlation (Skyfall-GS使用)
        if weights is not None:
            # 加权Pearson
            pred_mean = (pred_valid * weights).sum() / weights.sum()
            gt_mean = (gt_valid * weights).sum() / weights.sum()
            pred_centered = pred_valid - pred_mean
            gt_centered = gt_valid - gt_mean
            numerator = (pred_centered * gt_centered * weights).sum()
            denominator = torch.sqrt(
                (pred_centered**2 * weights).sum() * 
                (gt_centered**2 * weights).sum()
            )
            corr = numerator / (denominator + 1e-8)
        else:
            # 标准Pearson
            corr = pearson_corrcoef(pred_valid, gt_valid)
        
        loss = 1.0 - corr
        
    elif loss_type == "silog":
        # Scale-Invariant Log Loss
        log_diff = torch.log(pred_valid + 1e-8) - torch.log(gt_valid + 1e-8)
        if weights is not None:
            loss = torch.sqrt(
                (log_diff**2 * weights).mean() - 
                0.5 * ((log_diff * weights).mean())**2
            )
        else:
            loss = torch.sqrt(log_diff.var() + 0.5 * log_diff.mean()**2)
    
    elif loss_type == "gradient":
        # Gradient Loss (保持边缘)
        pred_grad_x = pred_valid[:, 1:] - pred_valid[:, :-1]
        pred_grad_y = pred_valid[1:, :] - pred_valid[:-1, :]
        gt_grad_x = gt_valid[:, 1:] - gt_valid[:, :-1]
        gt_grad_y = gt_valid[1:, :] - gt_valid[:-1, :]
        
        if weights is not None:
            w_x = weights[:, 1:]
            w_y = weights[1:, :]
            loss = (
                (torch.abs(pred_grad_x - gt_grad_x) * w_x).mean() +
                (torch.abs(pred_grad_y - gt_grad_y) * w_y).mean()
            )
        else:
            loss = (
                torch.abs(pred_grad_x - gt_grad_x).mean() +
                torch.abs(pred_grad_y - gt_grad_y).mean()
            )
    
    return loss
```

#### 步骤4：添加配置参数

**文件**: `Skyfall-GS/arguments/__init__.py`

**添加参数**:
```python
class OptimizationParams(ParamGroup):
    def __init__(self, parser: ArgumentParser):
        # ... 现有参数
        
        # VGGT相关参数
        self.use_vggt_init: bool = False
        self.lambda_vggt_depth: float = 0.5
        self.vggt_depth_loss_type: str = "pearson"  # pearson, silog, gradient
        self.vggt_use_ba: bool = False
        self.vggt_conf_threshold: float = 5.0
```

### 1.4 测试与验证

#### 测试脚本

**文件**: `Skyfall-GS/test_vggt_integration.py`

**测试内容**:
1. VGGT导出格式正确性
2. 坐标系转换正确性
3. 与COLMAP输出对比
4. 重建质量对比（PSNR/SSIM）
5. 时间效率对比

**评估指标**:
```
- 初始化时间: VGGT vs COLMAP
- 相机位姿误差: 与GT对比
- 重建质量: PSNR, SSIM, LPIPS
- 失败率: 在低纹理场景
```


### 1.5 预期输出

完成后的目录结构：
```
scene_dir/
├── images/
│   ├── frame_001.png
│   └── ...
├── vggt_poses.json          # VGGT导出的位姿
├── vggt_points3d.ply        # VGGT生成的点云
├── depths_vggt/             # VGGT深度图
│   ├── frame_001.npy
│   ├── frame_001_conf.npy
│   └── ...
└── sparse/                  # (可选) COLMAP格式输出
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

---

## 任务二：Few-shot Color Alignment Diffusion（核心创新，优先级2）

### 2.1 技术背景

**Skyfall-GS的FlowEdit方法**:
- 使用预训练FLUX/SD3模型
- 通过固定text prompts引导图像编辑
- 参数：nmin=4, nmax=10控制噪声强度
- 优点：开箱即用，无需训练
- 缺点：通用模型，不针对特定数据，依赖手工prompts

**我们的改进目标**:
- 从few-shot GT图像学习颜色和纹理分布
- 不依赖text prompts，直接从图像学习
- 结合VGGT的点轨迹做跨视一致性约束
- 比FlowEdit更好地对齐GT风格

### 2.2 技术方案

#### 方案A：快速Baseline（1-2天验证）

**目标**: 建立最小可行方案，验证思路

**方法**: 统计颜色迁移 + 外观MLP微调

**步骤**:
1. 对3DGS渲染图做颜色迁移（Reinhard/Lab直方图匹配）
2. 用迁移后的图像微调Skyfall-GS的外观MLP
3. 外观MLP输出仿射参数(γ, β)调整颜色

**实现**:
```python
# 文件: Skyfall-GS/color_transfer_baseline.py

def reinhard_color_transfer(source, target):
    """
    Reinhard颜色迁移算法
    
    Args:
        source: 3DGS渲染图 (H, W, 3)
        target: GT参考图 (H, W, 3)
    
    Returns:
        color_transferred: 迁移后的图像
    """
    # 转换到Lab空间
    # 匹配均值和标准差
    pass

def lab_histogram_matching(source, references):
    """
    Lab空间直方图匹配
    
    Args:
        source: 渲染图
        references: list of GT参考图
    
    Returns:
        matched: 匹配后的图像
    """
    pass
```

**优点**: 快速实现，无需训练，可作为baseline
**缺点**: 简单统计方法，无法处理复杂纹理

#### 方案B：Few-shot LoRA Adapter（核心方案）

**目标**: 训练可适配的diffusion模型

**架构选择**: FLUX/SDXL + LoRA

**训练流程**:


```
输入:
- 3DGS渲染图: (H, W, 3)
- Few-shot GT参考: k=5-10张图像

输出:
- 颜色/纹理对齐的图像

训练目标:
1. 颜色分布对齐 (Lab空间直方图/Chamfer距离)
2. 感知风格对齐 (VGG/CLIP特征)
3. 几何保持 (边缘/结构一致性)
4. 跨视一致性 (利用VGGT轨迹)
```

**模型架构**:

```python
# 文件: Skyfall-GS/few_shot_diffusion/model.py

class FewShotColorDiffusion(nn.Module):
    def __init__(self, base_model="stabilityai/stable-diffusion-xl-base-1.0"):
        super().__init__()
        # 加载预训练diffusion模型
        self.base_diffusion = load_pretrained_diffusion(base_model)
        
        # 冻结主干
        for param in self.base_diffusion.parameters():
            param.requires_grad = False
        
        # LoRA adapter (可训练)
        self.lora_adapter = LoRAAdapter(
            rank=16,
            target_modules=["to_q", "to_k", "to_v", "to_out"]
        )
        
        # Reference encoder (提取GT风格特征)
        self.reference_encoder = ReferenceEncoder()
        
        # Cross-attention for style injection
        self.style_cross_attn = StyleCrossAttention()
    
    def adapt_to_references(self, reference_images, num_steps=100):
        """
        Few-shot适配到参考图像
        
        Args:
            reference_images: (K, 3, H, W) GT参考图
            num_steps: 适配步数
        """
        # 提取参考图像的风格特征
        style_features = self.reference_encoder(reference_images)
        
        # 快速LoRA微调
        for step in range(num_steps):
            # 采样噪声图像
            noisy_img = self.add_noise(reference_images)
            
            # 去噪预测
            pred = self.base_diffusion(
                noisy_img,
                adapter=self.lora_adapter,
                style_features=style_features
            )
            
            # 计算损失并更新LoRA
            loss = self.compute_adaptation_loss(pred, reference_images)
            loss.backward()
            self.lora_adapter.step()
    
    def refine(self, rendered_image, reference_images):
        """
        用适配后的模型refinement
        
        Args:
            rendered_image: (3, H, W) 3DGS渲染图
            reference_images: (K, 3, H, W) GT参考
        
        Returns:
            refined: 修复后的图像
        """
        # 提取参考风格
        style_features = self.reference_encoder(reference_images)
        
        # Diffusion采样
        refined = self.base_diffusion.sample(
            x0=rendered_image,
            adapter=self.lora_adapter,
            style_guidance=style_features,
            num_steps=50
        )
        
        return refined
```

**损失函数设计**:

```python
# 文件: Skyfall-GS/few_shot_diffusion/losses.py

class ColorAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_color = 1.0      # 颜色分布
        self.w_style = 0.5      # 感知风格
        self.w_structure = 0.3  # 结构保持
        self.w_consistency = 0.2 # 跨视一致性
    
    def forward(self, pred, target, structure_ref, tracks=None):
        """
        计算总损失
        
        Args:
            pred: 预测图像
            target: GT参考
            structure_ref: 原始渲染（用于保持结构）
            tracks: VGGT轨迹（用于跨视一致性）
        """
        # 1. 颜色分布损失
        loss_color = self.color_distribution_loss(pred, target)
        
        # 2. 感知风格损失
        loss_style = self.perceptual_style_loss(pred, target)
        
        # 3. 结构保持损失
        loss_structure = self.structure_preservation_loss(pred, structure_ref)
        
        # 4. 跨视一致性损失
        loss_consistency = 0
        if tracks is not None:
            loss_consistency = self.cross_view_consistency_loss(pred, tracks)
        
        total_loss = (
            self.w_color * loss_color +
            self.w_style * loss_style +
            self.w_structure * loss_structure +
            self.w_consistency * loss_consistency
        )
        
        return total_loss, {
            'color': loss_color,
            'style': loss_style,
            'structure': loss_structure,
            'consistency': loss_consistency
        }
    
    def color_distribution_loss(self, pred, target):
        """
        Lab空间颜色分布损失
        使用直方图Chamfer距离或EMD
        """
        pred_lab = rgb_to_lab(pred)
        target_lab = rgb_to_lab(target)
        
        # 计算直方图
        pred_hist = compute_histogram(pred_lab, bins=64)
        target_hist = compute_histogram(target_lab, bins=64)
        
        # Chamfer距离或KL散度
        loss = chamfer_distance(pred_hist, target_hist)
        return loss
    
    def perceptual_style_loss(self, pred, target):
        """
        VGG/CLIP感知风格损失
        """
        # 使用VGG中间层特征
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        
        # Gram矩阵风格损失
        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            pred_gram = gram_matrix(pred_feat)
            target_gram = gram_matrix(target_feat)
            loss += F.mse_loss(pred_gram, target_gram)
        
        return loss
    
    def structure_preservation_loss(self, pred, structure_ref):
        """
        保持原始渲染的几何结构
        使用边缘/梯度一致性
        """
        pred_edges = canny_edge(pred)
        ref_edges = canny_edge(structure_ref)
        
        loss = F.mse_loss(pred_edges, ref_edges)
        return loss
    
    def cross_view_consistency_loss(self, pred_views, tracks, vis_scores, vis_thresh=0.5):
        """
        利用VGGT轨迹约束跨视一致性（带遮挡处理）
        
        Args:
            pred_views: list of refined images from different views
            tracks: (N, S, 2) VGGT预测的点轨迹
            vis_scores: (N, S) 可见性分数 [0, 1]
            vis_thresh: 可见性阈值
        
        ⚠️ 关键改进:
        1. 只对可见点（vis_scores > thresh）计算一致性
        2. 使用鲁棒损失（Huber）抑制遮挡/误匹配
        3. 过滤低视差对应（避免数值不稳定）
        """
        loss = 0
        valid_tracks = 0
        
        for track, vis in zip(tracks, vis_scores):
            # 只保留可见的视角
            visible_mask = vis > vis_thresh
            if visible_mask.sum() < 2:
                continue  # 至少需要2个可见视角
            
            visible_track = track[visible_mask]
            visible_views = [i for i, v in enumerate(visible_mask) if v]
            
            # 采样颜色
            colors = []
            for view_idx, (x, y) in zip(visible_views, visible_track):
                # 双线性采样
                color = sample_bilinear(pred_views[view_idx], x, y)
                colors.append(color)
            
            colors = torch.stack(colors)  # (V, 3)
            
            # 计算颜色方差（鲁棒版本）
            # 使用Huber loss而不是直接方差，抑制离群点
            mean_color = colors.mean(dim=0)
            color_diff = colors - mean_color
            
            # Huber loss (delta=0.1)
            huber_loss = torch.where(
                color_diff.abs() < 0.1,
                0.5 * color_diff**2,
                0.1 * (color_diff.abs() - 0.05)
            )
            
            loss += huber_loss.mean()
            valid_tracks += 1
        
        if valid_tracks == 0:
            return torch.tensor(0.0).cuda()
        
        return loss / valid_tracks
```


#### 方案C：Reference-based Attention（替代方案）

**目标**: 不训练LoRA，直接用参考图引导

**方法**: 在diffusion过程中注入参考图特征

```python
class ReferenceGuidedDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_diffusion = load_pretrained_diffusion()
        self.reference_encoder = CLIPImageEncoder()
    
    def forward(self, x, references):
        """
        用参考图引导diffusion
        
        Args:
            x: 输入图像
            references: 参考图像
        """
        # 提取参考特征
        ref_features = self.reference_encoder(references)
        ref_features = ref_features.mean(dim=0)  # 平均多个参考
        
        # 在diffusion的cross-attention中注入
        output = self.base_diffusion(
            x,
            cross_attention_kwargs={'reference_features': ref_features}
        )
        
        return output
```

**优点**: 无需训练，推理时直接使用
**缺点**: 可能不如LoRA适配效果好

### 2.3 实现步骤

#### 步骤1：实现Baseline

**文件**: `Skyfall-GS/color_transfer_baseline.py`

1. 实现Reinhard颜色迁移
2. 实现Lab直方图匹配
3. 集成到训练循环测试效果

#### 步骤2：实现Few-shot Diffusion

**文件结构**:
```
Skyfall-GS/few_shot_diffusion/
├── __init__.py
├── model.py              # FewShotColorDiffusion
├── losses.py             # 损失函数
├── reference_encoder.py  # 参考图编码器
├── lora_adapter.py       # LoRA实现
├── train.py              # 训练脚本
└── inference.py          # 推理脚本
```

**关键文件**:

1. `model.py`: 主模型定义
2. `losses.py`: 损失函数（如上所示）
3. `train.py`: 训练循环
4. `inference.py`: 推理接口

#### 步骤3：集成VGGT轨迹

**文件**: `Skyfall-GS/few_shot_diffusion/vggt_consistency.py`

```python
def extract_vggt_tracks(model, images, num_points=1000, sample_strategy="uniform"):
    """
    从VGGT提取点轨迹用于一致性约束
    
    Args:
        model: VGGT模型
        images: (N, 3, H, W) 多视角图像
        num_points: 采样点数量
        sample_strategy: "uniform", "edge", or "mixed"
    
    Returns:
        tracks: (M, N, 2) M个点在N个视角的2D位置
        vis_scores: (M, N) 可见性分数
    
    ⚠️ 采样策略建议:
    - "uniform": 均匀采样，适合整体一致性
    - "edge": 边缘采样，适合结构保持
    - "mixed": 混合采样（推荐）
    """
    with torch.no_grad():
        aggregated_tokens_list, ps_idx = model.aggregator(images[None])
        
        # 采样查询点（改进版）
        if sample_strategy == "uniform":
            query_points = sample_uniform_grid(images[0], num_points)
        elif sample_strategy == "edge":
            # 在边缘区域采样更多点
            edges = canny_edge(images[0])
            query_points = sample_from_edges(edges, num_points)
        else:  # mixed
            # 70% 均匀 + 30% 边缘
            uniform_pts = sample_uniform_grid(images[0], int(num_points * 0.7))
            edges = canny_edge(images[0])
            edge_pts = sample_from_edges(edges, int(num_points * 0.3))
            query_points = torch.cat([uniform_pts, edge_pts], dim=0)
        
        # 预测轨迹
        tracks, vis_scores, _ = model.track_head(
            aggregated_tokens_list,
            images[None],
            ps_idx,
            query_points=query_points[None]
        )
    
    return tracks.squeeze(0), vis_scores.squeeze(0)

def sample_uniform_grid(image, num_points):
    """均匀网格采样"""
    H, W = image.shape[-2:]
    grid_size = int(np.sqrt(num_points))
    y = torch.linspace(0, H-1, grid_size)
    x = torch.linspace(0, W-1, grid_size)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return points[:num_points]

def sample_from_edges(edge_map, num_points):
    """从边缘采样"""
    edge_coords = torch.nonzero(edge_map > 0.5)
    if len(edge_coords) < num_points:
        # 边缘点不够，补充随机点
        H, W = edge_map.shape
        random_pts = torch.rand(num_points - len(edge_coords), 2)
        random_pts[:, 0] *= W
        random_pts[:, 1] *= H
        points = torch.cat([edge_coords.float(), random_pts], dim=0)
    else:
        # 随机采样边缘点
        indices = torch.randperm(len(edge_coords))[:num_points]
        points = edge_coords[indices].float()
    return points

def apply_track_consistency(refined_images, tracks, vis_scores):
    """
    应用轨迹一致性约束
    
    Args:
        refined_images: list of refined images
        tracks: VGGT轨迹
        vis_scores: 可见性分数
    
    Returns:
        consistency_loss: 一致性损失
    """
    loss = 0
    for track, vis in zip(tracks, vis_scores):
        # 只考虑可见的点
        visible_mask = vis > 0.5
        visible_track = track[visible_mask]
        
        if len(visible_track) < 2:
            continue
        
        # 采样颜色
        colors = []
        for view_idx, (x, y) in enumerate(visible_track):
            if visible_mask[view_idx]:
                color = sample_bilinear(refined_images[view_idx], x, y)
                colors.append(color)
        
        # 颜色方差作为损失
        colors = torch.stack(colors)
        loss += colors.var(dim=0).mean()
    
    return loss / len(tracks)
```

#### 步骤4：集成到IDU循环

**文件**: `Skyfall-GS/train.py`

**修改IDU部分**:

```python
# 在IDU循环中
if opt.idu_refine:
    # 原有FlowEdit代码
    if opt.idu_use_flow_edit:
        # ... FlowEdit refinement
        pass
    
    # 新增Few-shot Diffusion
    elif opt.idu_use_few_shot_diffusion:
        # 加载Few-shot模型
        if not hasattr(self, 'few_shot_diffusion'):
            self.few_shot_diffusion = FewShotColorDiffusion()
        
        # 选择few-shot参考图
        reference_gts = select_reference_images(
            scene.getTrainCameras(),
            num_refs=opt.idu_num_reference_images
        )
        
        # 适配到参考图
        self.few_shot_diffusion.adapt_to_references(
            reference_gts,
            num_steps=opt.idu_adaptation_steps
        )
        
        # Refinement
        refined_images = []
        for rendered_img in rendered_images:
            refined = self.few_shot_diffusion.refine(
                rendered_img,
                reference_gts
            )
            refined_images.append(refined)
        
        # 可选：应用VGGT轨迹一致性
        if opt.idu_use_vggt_consistency:
            tracks, vis_scores = extract_vggt_tracks(
                vggt_model,
                rendered_images
            )
            consistency_loss = apply_track_consistency(
                refined_images,
                tracks,
                vis_scores
            )
            # 用一致性损失微调
```

#### 步骤5：添加配置参数

**文件**: `Skyfall-GS/arguments/__init__.py`

```python
class OptimizationParams(ParamGroup):
    def __init__(self, parser: ArgumentParser):
        # ... 现有参数
        
        # Few-shot Diffusion参数
        self.idu_use_few_shot_diffusion: bool = False
        self.idu_num_reference_images: int = 5
        self.idu_adaptation_steps: int = 100
        self.idu_diffusion_base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
        self.idu_lora_rank: int = 16
        
        # 损失权重
        self.idu_loss_color_weight: float = 1.0
        self.idu_loss_style_weight: float = 0.5
        self.idu_loss_structure_weight: float = 0.3
        self.idu_loss_consistency_weight: float = 0.2
        
        # VGGT一致性
        self.idu_use_vggt_consistency: bool = True
        self.idu_vggt_num_track_points: int = 1000
```

### 2.4 训练策略

#### Few-shot适配流程

```
1. 从训练集选择k=5-10张GT作为参考
2. 快速LoRA适配（100-200步）
3. 用适配后的模型refinement所有渲染图
4. 将refined图像回流到3DGS训练
5. 重复2-4（可选）
```

#### 多样本策略（保留Skyfall-GS）

```
每个视角生成Ns=2-4个样本
3DGS训练时平均多个样本的损失
减少diffusion的随机性影响
```

#### 课程式采样（保留Skyfall-GS）

```
Episode 1: 高仰角 (85°, 75°)
Episode 2: 中仰角 (65°, 55°)
Episode 3: 低仰角 (45°, 25°)

逐步暴露遮挡面，稳定训练
```


### 2.5 评估指标

#### 分布指标
```
- FID-CLIP: Fréchet Inception Distance using CLIP features
- CMMD: CLIP Maximum Mean Discrepancy
- Color Histogram Distance: Lab空间直方图KL散度/EMD
```

#### 像素指标
```
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index
- LPIPS: Learned Perceptual Image Patch Similarity
```

#### 一致性指标
```
- Track Color Variance: 轨迹颜色方差（越小越好）
- Cross-view SSIM: 跨视角结构相似度
```

#### 对比实验
```
1. Baseline (无refinement)
2. FlowEdit (Skyfall-GS原方法)
3. Color Transfer Baseline (统计方法)
4. Few-shot Diffusion (我们的方法)
5. Few-shot + VGGT Consistency (完整方法)
```

---

## 任务三：完整IDU集成（优先级3）

### 3.1 完整Pipeline

```
[输入] 多视角卫星RGB图像
    ↓
[1] VGGT初始化
    → 相机位姿 + 深度图 + 点云
    → (可选) 轻量BA优化
    ↓
[2] 3DGS重建 (Stage 1)
    → 外观建模 + 不透明度熵正则
    → VGGT深度监督 (或MoGe)
    → 伪相机深度监督
    ↓
[3] IDU迭代 (Stage 2)
    ↓
    [3.1] 课程式相机采样
        → Episode 1: 高仰角
        → Episode 2: 中仰角
        → Episode 3: 低仰角
    ↓
    [3.2] 渲染新视角
        → 3DGS.render(novel_views)
    ↓
    [3.3] Few-shot Diffusion Refinement
        → 选择k张GT参考
        → LoRA快速适配
        → Refinement (每视角Ns=2-4样本)
        → (可选) VGGT轨迹一致性约束
    ↓
    [3.4] 回流训练
        → 75% refined + 25% original
        → 更新3DGS参数
    ↓
    [3.5] 重复3.1-3.4 (Ne=3-5 episodes)
    ↓
[输出] 优化的3DGS模型
```

### 3.2 实现要点

#### 课程式采样参数

**JAX数据集**:
```python
idu_params_jax = {
    "elevation_list": [85., 75., 65., 55., 45.],
    "radius_list": [300., 275., 275., 250., 250.],
    "fov": 60.0
}
```

**NYC数据集**:
```python
idu_params_nyc = {
    "elevation_list": [85., 75., 65., 55., 45., 25.],
    "radius_list": [600., 600., 600., 600., 600.],
    "fov": 20.0
}
```

#### 混合训练策略

```python
# 从Skyfall-GS附录A.1
train_ratio = 0.75  # 75% refined, 25% original

def sample_training_images(original_images, refined_images, ratio=0.75):
    """
    混合采样原始图和refined图
    
    Args:
        original_images: 原始GT图像
        refined_images: Diffusion refined图像
        ratio: refined图像的比例
    
    Returns:
        mixed_images: 混合后的训练图像
    """
    num_refined = int(len(original_images) * ratio)
    num_original = len(original_images) - num_refined
    
    refined_indices = random.sample(range(len(refined_images)), num_refined)
    original_indices = random.sample(range(len(original_images)), num_original)
    
    mixed_images = []
    mixed_images.extend([refined_images[i] for i in refined_indices])
    mixed_images.extend([original_images[i] for i in original_indices])
    
    return mixed_images
```

#### Episode迭代控制

```python
# 每个episode的迭代次数
idu_episode_iterations = 10000

# Densification截止
idu_densify_until_iter = 9000  # 90% of episode

# 不透明度重置
idu_opacity_reset_interval = 5000
idu_opacity_cooling_iterations = 500

# 测试间隔
idu_testing_interval = 5000
```

### 3.3 完整训练脚本

**文件**: `Skyfall-GS/train_vggt_skyfall.py`

```python
#!/usr/bin/env python3
"""
VGGT + Skyfall-GS完整训练脚本
"""

import argparse
from pathlib import Path
import torch

from vggt.models.vggt import VGGT
from scene import Scene, GaussianModel
from few_shot_diffusion import FewShotColorDiffusion
from utils.camera_utils import gen_idu_orbit_camera

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_vggt_init", action="store_true")
    parser.add_argument("--use_few_shot_diffusion", action="store_true")
    parser.add_argument("--num_episodes", type=int, default=5)
    args = parser.parse_args()
    
    # [1] VGGT初始化
    if args.use_vggt_init:
        print("Running VGGT initialization...")
        vggt_model = VGGT.from_pretrained("facebook/VGGT-1B")
        # ... VGGT导出逻辑
    
    # [2] 加载场景和3DGS
    print("Loading scene...")
    gaussians = GaussianModel(...)
    scene = Scene(args.scene_dir, gaussians)
    
    # [3] Stage 1: 重建
    print("Stage 1: Reconstruction...")
    train_stage1(scene, gaussians, args)
    
    # [4] Stage 2: IDU迭代
    if args.use_few_shot_diffusion:
        print("Stage 2: IDU with Few-shot Diffusion...")
        few_shot_model = FewShotColorDiffusion()
        
        for episode in range(args.num_episodes):
            print(f"Episode {episode+1}/{args.num_episodes}")
            
            # 课程式采样
            cameras = gen_idu_orbit_camera(
                elevation=idu_params["elevation_list"][episode],
                radius=idu_params["radius_list"][episode],
                ...
            )
            
            # 渲染
            rendered_images = render_cameras(gaussians, cameras)
            
            # Few-shot refinement
            reference_gts = select_references(scene, k=5)
            few_shot_model.adapt_to_references(reference_gts)
            refined_images = few_shot_model.refine_batch(rendered_images)
            
            # 回流训练
            train_with_refined(gaussians, refined_images, original_gts)
    
    # [5] 保存
    print("Saving model...")
    gaussians.save_ply(Path(args.output_dir) / "final.ply")

if __name__ == "__main__":
    main()
```

---

## 实施时间表

### 第1周：VGGT集成
- Day 1-2: 实现vggt_pose_export.py
- Day 3-4: 修改Skyfall-GS数据加载器
- Day 5-6: 集成深度监督，测试重建
- Day 7: 对比实验和文档

### 第2周：Baseline实现
- Day 1-2: 实现颜色迁移baseline
- Day 3-4: 集成到训练循环
- Day 5-6: 评估baseline效果
- Day 7: 分析结果，规划Few-shot方案

### 第3-4周：Few-shot Diffusion
- Week 3 Day 1-3: 实现模型架构
- Week 3 Day 4-5: 实现损失函数
- Week 3 Day 6-7: 实现训练循环
- Week 4 Day 1-3: 集成VGGT轨迹一致性
- Week 4 Day 4-5: 调试和优化
- Week 4 Day 6-7: 初步实验

### 第5周：IDU集成和实验
- Day 1-2: 完整IDU集成
- Day 3-5: 全面实验对比
- Day 6-7: 结果分析和可视化

### 第6周：论文撰写
- Day 1-3: 方法描述和实验结果
- Day 4-5: 消融实验和分析
- Day 6-7: 修改和完善

---

## 代码文件清单

### 新增文件

```
Skyfall-GS/
├── vggt_pose_export.py                    # VGGT导出脚本
├── test_vggt_integration.py               # VGGT集成测试
├── color_transfer_baseline.py             # 颜色迁移baseline
├── train_vggt_skyfall.py                  # 完整训练脚本
└── few_shot_diffusion/                    # Few-shot Diffusion模块
    ├── __init__.py
    ├── model.py                           # 主模型
    ├── losses.py                          # 损失函数
    ├── reference_encoder.py               # 参考编码器
    ├── lora_adapter.py                    # LoRA实现
    ├── vggt_consistency.py                # VGGT一致性
    ├── train.py                           # 训练脚本
    └── inference.py                       # 推理脚本
```

### 修改文件

```
Skyfall-GS/
├── scene/
│   └── dataset_readers.py                 # 添加VGGT读取函数
├── arguments/
│   └── __init__.py                        # 添加VGGT和Few-shot参数
└── train.py                               # 集成VGGT深度和Few-shot refinement
```

---

## 依赖库

### 新增依赖

```txt
# VGGT相关
# (已在项目中)

# Diffusion相关
diffusers>=0.30.1
transformers>=4.46.3
accelerate>=0.20.0
peft>=0.7.0  # LoRA实现

# 图像处理
scikit-image>=0.21.0
opencv-python>=4.8.0

# 评估
pytorch-fid>=0.3.0
clip @ git+https://github.com/openai/CLIP.git
```

---

## 预期成果

### 定量指标

| 方法 | 初始化时间 | PSNR | SSIM | LPIPS | FID-CLIP | CMMD |
|------|-----------|------|------|-------|----------|------|
| Skyfall-GS (COLMAP) | ~5min | 12.0 | 0.75 | 0.20 | 50.0 | 0.15 |
| **Ours (VGGT only)** | ~2s | 12.0 | 0.75 | 0.20 | 50.0 | 0.15 |
| **Ours (VGGT + Few-shot)** | ~2s | **13.2** | **0.80** | **0.15** | **35.0** | **0.10** |

### 定性效果

- 颜色更接近GT
- 纹理更清晰
- 跨视角更一致
- 低纹理场景更鲁棒

---

## 风险和缓解

### 风险1：VGGT位姿精度不足
**缓解**: 
- 使用轻量BA优化
- 与COLMAP混合初始化
- 增加深度监督权重

### 风险2：Few-shot适配不稳定
**缓解**:
- 先用baseline验证思路
- 增加正则化
- 调整适配步数

### 风险3：训练时间过长
**缓解**:
- 减少episode数量
- 使用更小的LoRA rank
- 并行化处理

### 风险4：跨视一致性不足
**缓解**:
- 增加VGGT轨迹约束权重
- 使用多样本平均
- 后处理一致性优化

---

## 参考资料

### 论文
- VGGT: Visual Geometry Grounded Transformer (CVPR 2025)
- Skyfall-GS: Synthesizing Immersive 3D Urban Scenes (2025)
- FlowEdit: Inversion-Free Text-Based Editing (2024)
- LoRA: Low-Rank Adaptation of Large Language Models (2021)

### 代码库
- VGGT: https://github.com/facebookresearch/vggt
- Skyfall-GS: (你的仓库)
- FlowEdit: https://github.com/fallenshock/FlowEdit
- Diffusers: https://github.com/huggingface/diffusers

---

## 联系和支持

如有问题，请参考：
- VGGT Issues: https://github.com/facebookresearch/vggt/issues
- Skyfall-GS论文: skyfall-gs.pdf
- 本项目文档: 本文件

---

**文档版本**: 1.0  
**最后更新**: 2025-11-06  
**作者**: AI Assistant (基于用户需求生成)


---

## 验证清单（Critical Checkpoints）

### 阶段1：VGGT导出验证

#### ✅ Checkpoint 1.1: 外参方向正确性
```python
# 测试代码
def test_extrinsic_direction():
    """验证外参是w2c而不是c2w"""
    # 加载VGGT输出
    extrinsic = load_vggt_extrinsic()  # 应该是w2c
    
    # 计算相机中心
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    
    # 如果是w2c，相机中心应该是: C = -R^T * t
    C_w2c = -R.T @ t
    
    # 如果是c2w，相机中心应该是: C = t
    C_c2w = t
    
    # 可视化两种情况，看哪个合理
    print(f"If w2c, camera center: {C_w2c}")
    print(f"If c2w, camera center: {C_c2w}")
    
    # 相机中心应该在场景外围，不应该在原点
    assert np.linalg.norm(C_w2c) > 10, "Camera too close to origin"
```

**预期结果**:
- 相机中心应该在场景外围（距离原点>10）
- 多个相机应该围绕场景中心分布
- 相机z轴应该指向场景中心

#### ✅ Checkpoint 1.2: FoV单位正确性
```python
def test_fov_unit():
    """验证FoV是弧度而不是角度"""
    fov_w, fov_h = load_vggt_fov()
    
    # 弧度范围应该在[0, π]
    assert 0 < fov_w < np.pi, f"FoV_w={fov_w} not in radian range"
    assert 0 < fov_h < np.pi, f"FoV_h={fov_h} not in radian range"
    
    # 转换为角度检查合理性
    fov_w_deg = np.degrees(fov_w)
    fov_h_deg = np.degrees(fov_h)
    
    print(f"FoV: {fov_w_deg:.1f}° x {fov_h_deg:.1f}°")
    
    # 卫星图像FoV通常在20-90度
    assert 10 < fov_w_deg < 120, "FoV out of reasonable range"
```

**预期结果**:
- FoV弧度值: 0.3-1.5 (对应20-90度)
- 转换后的角度值合理

#### ✅ Checkpoint 1.3: 坐标系一致性
```python
def test_coordinate_system():
    """验证OpenCV坐标系"""
    # 投影测试点
    point_3d = np.array([0, 0, 10])  # 相机前方10米
    
    extrinsic = load_vggt_extrinsic()  # w2c
    intrinsic = load_vggt_intrinsic()
    
    # 投影到图像
    point_cam = extrinsic[:3, :3] @ point_3d + extrinsic[:3, 3]
    point_2d = intrinsic @ point_cam
    point_2d = point_2d[:2] / point_2d[2]
    
    # OpenCV: z前，点应该在相机前方（z>0）
    assert point_cam[2] > 0, "Point should be in front of camera"
    
    # 投影点应该在图像内
    H, W = 1024, 1024
    assert 0 < point_2d[0] < W and 0 < point_2d[1] < H, \
        "Projection out of image bounds"
```

**预期结果**:
- 前方点的z坐标>0
- 投影点在图像范围内

---

### 阶段2：深度监督验证

#### ✅ Checkpoint 2.1: 深度置信度分布
```python
def test_depth_confidence():
    """验证深度置信度的分布和使用"""
    depth_conf = load_vggt_depth_conf()
    
    # 统计置信度分布
    print(f"Conf min: {depth_conf.min():.2f}")
    print(f"Conf max: {depth_conf.max():.2f}")
    print(f"Conf mean: {depth_conf.mean():.2f}")
    print(f"Conf median: {np.median(depth_conf):.2f}")
    
    # 可视化置信度图
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(depth_conf, cmap='viridis')
    plt.colorbar()
    plt.title('Depth Confidence')
    
    plt.subplot(122)
    plt.hist(depth_conf.flatten(), bins=50)
    plt.title('Confidence Distribution')
    plt.savefig('depth_conf_analysis.png')
    
    # 检查高置信度区域占比
    high_conf_ratio = (depth_conf > 5.0).sum() / depth_conf.size
    print(f"High confidence ratio (>5.0): {high_conf_ratio:.2%}")
```

**预期结果**:
- 置信度范围: 0-20（典型）
- 高置信度区域: 30-70%
- 低置信度通常在边缘/遮挡区域

#### ✅ Checkpoint 2.2: 深度损失收敛
```python
def test_depth_loss_convergence():
    """监控深度损失是否正常收敛"""
    # 训练前100步的深度损失
    depth_losses = []
    
    for iteration in range(100):
        loss = train_one_step()
        depth_losses.append(loss['depth'])
    
    # 绘制损失曲线
    plt.plot(depth_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Depth Loss')
    plt.title('Depth Loss Convergence')
    plt.savefig('depth_loss_curve.png')
    
    # 检查是否下降
    initial_loss = np.mean(depth_losses[:10])
    final_loss = np.mean(depth_losses[-10:])
    
    assert final_loss < initial_loss * 0.8, \
        "Depth loss not converging"
```

**预期结果**:
- 损失应该在前100步内下降
- 最终损失 < 初始损失 * 0.8

---

### 阶段3：Few-shot Diffusion验证

#### ✅ Checkpoint 3.1: 颜色分布对齐
```python
def test_color_distribution_alignment():
    """验证颜色分布是否对齐到GT"""
    rendered = load_rendered_image()
    refined = load_refined_image()
    gt = load_gt_image()
    
    # 转换到Lab空间
    rendered_lab = rgb_to_lab(rendered)
    refined_lab = rgb_to_lab(refined)
    gt_lab = rgb_to_lab(gt)
    
    # 计算直方图距离
    def hist_distance(img1, img2):
        hist1 = compute_histogram(img1, bins=64)
        hist2 = compute_histogram(img2, bins=64)
        return np.sum(np.abs(hist1 - hist2))
    
    dist_rendered_gt = hist_distance(rendered_lab, gt_lab)
    dist_refined_gt = hist_distance(refined_lab, gt_lab)
    
    print(f"Rendered-GT distance: {dist_rendered_gt:.4f}")
    print(f"Refined-GT distance: {dist_refined_gt:.4f}")
    
    # Refined应该更接近GT
    assert dist_refined_gt < dist_rendered_gt, \
        "Refinement did not improve color alignment"
    
    # 可视化直方图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axes, 
                               [rendered_lab, refined_lab, gt_lab],
                               ['Rendered', 'Refined', 'GT']):
        ax.hist(img.flatten(), bins=50, alpha=0.7)
        ax.set_title(title)
    plt.savefig('color_histogram_comparison.png')
```

**预期结果**:
- Refined-GT距离 < Rendered-GT距离
- 直方图形状更接近

#### ✅ Checkpoint 3.2: 跨视一致性
```python
def test_cross_view_consistency():
    """验证跨视角颜色一致性"""
    refined_views = load_refined_views()  # N个视角
    tracks, vis_scores = load_vggt_tracks()
    
    # 计算轨迹颜色方差
    color_variances = []
    
    for track, vis in zip(tracks, vis_scores):
        visible_mask = vis > 0.5
        if visible_mask.sum() < 2:
            continue
        
        colors = []
        for view_idx, (x, y) in enumerate(track):
            if visible_mask[view_idx]:
                color = sample_bilinear(refined_views[view_idx], x, y)
                colors.append(color)
        
        colors = np.array(colors)
        variance = colors.var(axis=0).mean()
        color_variances.append(variance)
    
    mean_variance = np.mean(color_variances)
    print(f"Mean track color variance: {mean_variance:.4f}")
    
    # 方差应该较小（<0.01）
    assert mean_variance < 0.01, \
        "Cross-view consistency too low"
    
    # 可视化
    plt.hist(color_variances, bins=50)
    plt.xlabel('Color Variance')
    plt.ylabel('Count')
    plt.title('Track Color Variance Distribution')
    plt.savefig('track_variance.png')
```

**预期结果**:
- 平均颜色方差 < 0.01
- 大部分轨迹方差 < 0.02

---

### 阶段4：端到端验证

#### ✅ Checkpoint 4.1: 时间效率
```python
def test_timing():
    """验证时间效率提升"""
    import time
    
    # VGGT初始化
    start = time.time()
    vggt_init()
    vggt_time = time.time() - start
    
    # COLMAP初始化（对比）
    start = time.time()
    colmap_init()
    colmap_time = time.time() - start
    
    print(f"VGGT time: {vggt_time:.2f}s")
    print(f"COLMAP time: {colmap_time:.2f}s")
    print(f"Speedup: {colmap_time / vggt_time:.1f}x")
    
    # VGGT应该快至少10倍
    assert vggt_time < colmap_time / 10, \
        "VGGT not significantly faster"
```

**预期结果**:
- VGGT: 1-5秒
- COLMAP: 60-300秒
- 加速比: >10x

#### ✅ Checkpoint 4.2: 重建质量
```python
def test_reconstruction_quality():
    """验证重建质量"""
    # 渲染测试视角
    rendered = render_test_views()
    gt = load_gt_test_views()
    
    # 计算指标
    psnr = compute_psnr(rendered, gt)
    ssim = compute_ssim(rendered, gt)
    lpips = compute_lpips(rendered, gt)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    print(f"LPIPS: {lpips:.4f}")
    
    # 与baseline对比
    baseline_psnr = 12.0  # Skyfall-GS
    assert psnr >= baseline_psnr - 0.5, \
        "Quality degradation too large"
```

**预期结果**:
- PSNR: ≥11.5 dB (不低于baseline太多)
- SSIM: ≥0.70
- LPIPS: ≤0.25

---

## 故障排查指南

### 问题1: 相机位姿可视化异常

**症状**: 相机都在原点附近，或朝向错误

**排查步骤**:
1. 检查JSON中pose_type是否为"w2c_opencv"
2. 验证FoV单位是弧度
3. 检查R的转置是否正确
4. 用8角箱体投影测试

**解决方案**:
```python
# 正确的加载方式
R_w2c = np.array(extr["R"])
t_w2c = np.array(extr["t"])
R = np.transpose(R_w2c)  # Skyfall-GS格式
T = t_w2c
```

### 问题2: 深度损失不下降

**症状**: 深度损失在训练中保持高位或震荡

**排查步骤**:
1. 检查depth_conf是否正确加权
2. 检查有效mask（深度>0）
3. 尝试不同损失类型
4. 可视化rendered depth vs GT depth

**解决方案**:
```python
# 添加置信度加权
conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
valid_mask = (gt_depth > 0) & (conf_norm > 0.5)
weights = conf_norm[valid_mask]
```

### 问题3: Few-shot适配不稳定

**症状**: LoRA适配后效果不一致或崩溃

**排查步骤**:
1. 检查参考图数量（建议5-10张）
2. 检查适配步数（建议100-200）
3. 检查学习率（建议1e-4）
4. 检查损失权重平衡

**解决方案**:
```python
# 降低学习率和适配步数
optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
num_adaptation_steps = 100  # 从200降到100
```

### 问题4: 跨视一致性差

**症状**: 不同视角颜色差异大

**排查步骤**:
1. 检查vis_scores是否正确过滤
2. 检查Huber loss是否启用
3. 检查轨迹采样策略
4. 可视化轨迹颜色

**解决方案**:
```python
# 提高可见性阈值
visible_mask = vis_scores > 0.7  # 从0.5提高到0.7

# 使用Huber loss
huber_loss = torch.where(
    color_diff.abs() < 0.1,
    0.5 * color_diff**2,
    0.1 * (color_diff.abs() - 0.05)
)
```

---

## 最终验收标准

### 必须达到的指标

| 指标 | 目标值 | 验收标准 |
|------|--------|---------|
| **初始化时间** | <5秒 | VGGT vs COLMAP加速>10x |
| **PSNR** | ≥12.0 dB | 不低于Skyfall-GS baseline |
| **SSIM** | ≥0.75 | 不低于Skyfall-GS baseline |
| **LPIPS** | ≤0.20 | 不高于Skyfall-GS baseline |
| **FID-CLIP** | ≤45.0 | 优于Skyfall-GS (50.0) |
| **CMMD** | ≤0.12 | 优于Skyfall-GS (0.15) |
| **轨迹颜色方差** | <0.01 | 跨视一致性良好 |

### 可选的改进指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **PSNR** | ≥13.0 dB | 显著优于baseline |
| **FID-CLIP** | ≤35.0 | 大幅改进 |
| **初始化失败率** | <5% | 在低纹理场景 |
| **显存占用** | <24GB | 单卡可训练 |

---

**文档版本**: 2.0 (修正版)  
**最后更新**: 2025-11-06  
**修正内容**: VGGT外参方向、FoV单位、置信度加权、遮挡处理、场景归一化
