# 模型对比工具使用指南

## 🎯 目的
对比 fine-tuned 模型和 baseline 模型在单 pair 数据上的深度预测效果。

## 📋 修复内容

### 1. 模型加载问题
**问题**: Fine-tuned checkpoint 包含完整训练状态，而不仅仅是模型权重。

**解决方案**:
- 正确提取 `checkpoint['model']` 中的模型权重
- 移除 DDP 训练产生的 `module.` 前缀
- 使用与训练时相同的模型配置（`enable_point=False`, `enable_track=False`）

### 2. 深度值可视化策略
**策略**: 每个深度图独立归一化到 0-1 进行可视化

**原因**:
- 不同模型预测的深度值范围可能差异很大
- 独立归一化可以更清楚地看到每个模型预测的细节和结构
- 在标题中显示每个深度图的实际值范围，便于数值对比

**实现**:
- GT、Fine-tuned、Baseline 各自独立归一化
- 每个深度图的 min 值映射到 0（紫色）
- 每个深度图的 max 值映射到 1（黄色）
- 标题显示原始深度范围（单位：米）

## 🚀 使用方法

### 步骤 1: 测试模型加载
```bash
python test_model_loading.py
```

预期输出：
```
✅ Baseline model loaded
   Missing keys: 0 (或少量，对应禁用的 heads)
   Unexpected keys: 0

✅ Fine-tuned model loaded
   Missing keys: 0 (或少量，对应禁用的 heads)
   Unexpected keys: 0 (或少量，对应额外的训练层)
```

### 步骤 2: 运行模型对比（推荐使用脚本）
```bash
./run_comparison.sh
```

或者直接运行 Python：
```bash
python compare_models.py \
    --data_dir training/dataset_aerialmd_single/cropped \
    --finetuned_model training/logs/single_pair_test/ckpts/checkpoint.pt \
    --baseline_model model/vggt_1B_commercial.pt \
    --output model_comparison
```

**重要**: 脚本会分 3 个阶段运行，避免显存不足：
1. **阶段 1**: 加载 fine-tuned 模型，预测所有图像，然后释放显存
2. **阶段 2**: 加载 baseline 模型，预测所有图像，然后释放显存
3. **阶段 3**: 加载 GT 和 RGB，生成对比可视化

### 步骤 3: 查看结果
```bash
ls -lh model_comparison/
```

## 📊 输出说明

### 1. 可视化对比图
每张图像生成一个 4 列对比图：

```
[RGB Image] | [Ground Truth] | [Fine-tuned] | [Baseline]
                   ↓                 ↓              ↓
              Valid pixels      MAE/RMSE      MAE/RMSE
                                Abs Rel       Abs Rel
```

**文件命名**: `{scene}_{image}_comparison.png`

**关键特性**:
- 每个深度图独立归一化到 0-1（充分利用色彩范围）
- 标题显示实际深度范围（单位：米）
- 图像下方显示深度指标（MAE、RMSE、Abs Rel）
- 图底部中央显示改进百分比
- 无效区域显示为黑色

### 2. 定量指标

#### 可视化图中的指标
每个预测深度图下方显示：
- **MAE** (Mean Absolute Error): 平均绝对误差（米）
- **RMSE** (Root Mean Square Error): 均方根误差（米）
- **Abs Rel** (Absolute Relative Error): 绝对相对误差（无量纲）

图底部中央显示：
- **Improvement**: Fine-tuned 相对 Baseline 的改进百分比

#### 控制台输出
```
Fine-tuned Model:
  Average MAE: 0.234
  Average RMSE: 0.456

Baseline Model:
  Average MAE: 3.567
  Average RMSE: 5.789

Improvement:
  MAE: +93.44%
  RMSE: +92.12%
```

详细的指标说明请参考 `METRICS_EXPLANATION.md`

## ✅ 成功标准

### 过拟合成功的表现：

1. **视觉对比**:
   - Fine-tuned 深度图与 GT 颜色分布非常相似
   - Baseline 深度图与 GT 有明显差异
   - 细节和边界更清晰

2. **定量指标**:
   - MAE 改进 > 80%
   - RMSE 改进 > 80%
   - Fine-tuned MAE < 0.5 米
   - Baseline MAE > 2 米

### 需要调试的情况：

1. **改进 < 20%**: 训练可能没有收敛
2. **Fine-tuned 与 Baseline 相似**: 检查是否真的加载了 fine-tuned 权重
3. **预测结果异常**: 检查模型配置是否与训练时一致

## 🔧 关键代码逻辑

### 0. 显存优化策略
```python
# 阶段 1: Fine-tuned 模型
finetuned_model = load_model_from_pt(args.finetuned_model, device)
for data in image_data:
    data['finetuned_depth'] = predict_depth(finetuned_model, data['img_path'], device, dtype)
del finetuned_model
torch.cuda.empty_cache()  # 释放显存

# 阶段 2: Baseline 模型
baseline_model = load_model_from_pt(args.baseline_model, device)
for data in image_data:
    data['baseline_depth'] = predict_depth(baseline_model, data['img_path'], device, dtype)
del baseline_model
torch.cuda.empty_cache()  # 释放显存

# 阶段 3: 生成对比图（不需要模型）
for data in image_data:
    create_comparison_figure(...)
```

### 1. 模型加载
```python
# 提取模型权重
checkpoint = torch.load(model_path, map_location=device)
if 'model' in checkpoint:
    state_dict = checkpoint['model']  # 训练 checkpoint
else:
    state_dict = checkpoint  # 纯模型权重

# 移除 DDP 前缀
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v

# 加载（允许部分匹配）
model.load_state_dict(new_state_dict, strict=False)
```

### 2. 深度预测
```python
# 分别运行两个模型
finetuned_depth = predict_depth(finetuned_model, image_path, device, dtype)
baseline_depth = predict_depth(baseline_model, image_path, device, dtype)
```

### 3. 独立归一化可视化
```python
# 每个深度图独立归一化到 0-1
gt_colored, gt_min, gt_max = colorize_depth(gt_depth)
ft_colored, ft_min, ft_max = colorize_depth(finetuned_depth)
bl_colored, bl_min, bl_max = colorize_depth(baseline_depth)

# 显示原始深度范围
print(f"GT: [{gt_min:.2f}, {gt_max:.2f}]")
print(f"Fine-tuned: [{ft_min:.2f}, {ft_max:.2f}]")
print(f"Baseline: [{bl_min:.2f}, {bl_max:.2f}]")
```

**可视化效果**:
- 每个深度图都充分利用整个色彩范围（viridis: 紫→绿→黄）
- 更容易看到细节和结构
- 通过标题中的数值范围进行定量对比

## 📝 注意事项

1. **模型配置一致性**: 推理时必须使用与训练时相同的模型配置
2. **独立归一化**: 每个深度图独立归一化到 0-1，便于看清细节
3. **数值对比**: 通过标题中的实际深度范围进行定量对比
4. **无效区域**: 深度值 ≤ 0 的区域显示为黑色
5. **内存管理**: 使用 `torch.no_grad()` 和 `torch.cuda.amp.autocast()` 节省内存

## 🎨 可视化说明

### 色彩映射（Viridis）
- **紫色**: 最小深度值（近处）
- **绿色**: 中等深度值
- **黄色**: 最大深度值（远处）

### 独立归一化的优势
- ✅ 每个深度图都充分利用色彩范围
- ✅ 更容易看到预测的细节和结构
- ✅ 避免某个模型的预测被"压缩"到很小的色彩范围
- ⚠️ 需要通过标题中的数值范围进行定量对比

### 示例解读
```
Ground Truth: [20.82, 35.76]  → 实际深度 20-36 米
Fine-tuned:   [0.01, 2.13]    → 预测深度 0-2 米（明显偏小）
Baseline:     [0.59, 1.76]    → 预测深度 0.6-1.8 米（明显偏小）
```
这说明两个模型都严重低估了深度值，需要检查训练配置。

## 🐛 故障排除

### 问题 1: Missing keys 错误
```
Missing key(s) in state_dict: "point_head.xxx", "track_head.xxx"
```
**原因**: 训练时禁用了这些 heads
**解决**: 正常现象，使用 `strict=False` 即可

### 问题 2: Unexpected keys 错误
```
Unexpected key(s): "depth_head.compress_vit_xxx"
```
**原因**: 训练时使用了额外的层
**解决**: 正常现象，使用 `strict=False` 即可

### 问题 3: 深度图全黑
**可能原因**:
- 模型输出全零
- 深度范围异常
- 数据预处理问题

**调试步骤**:
1. 打印深度图的统计信息（min, max, mean）
2. 检查模型是否正确加载
3. 检查输入图像预处理

---

**创建时间**: 2025-10-19  
**目的**: 验证单 pair 过拟合效果  
**预期**: Fine-tuned 模型显著优于 baseline
