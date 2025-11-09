# 深度图可视化策略说明

## 🎯 可视化目标

在对比 fine-tuned 模型和 baseline 模型时，我们需要：
1. **看清细节**: 每个模型预测的结构和细节
2. **定量对比**: 实际深度值的差异
3. **视觉对比**: 预测质量的直观比较

## 🔄 两种可视化策略对比

### 策略 1: 统一深度范围（已弃用）

```python
# 使用 GT 的深度范围统一所有深度图
vmin, vmax = np.percentile(gt_depth[gt_depth > 0], [2, 98])

gt_colored = colorize_depth(gt_depth, vmin, vmax)
ft_colored = colorize_depth(finetuned_depth, vmin, vmax)
bl_colored = colorize_depth(baseline_depth, vmin, vmax)
```

**优点**:
- ✅ 色彩直接对应相同的深度值
- ✅ 视觉上可以直接比较深度

**缺点**:
- ❌ 如果模型预测范围与 GT 差异很大，预测图会很暗或很亮
- ❌ 无法看清预测的细节和结构
- ❌ 不适合调试阶段（模型可能预测错误的深度范围）

**示例问题**:
```
GT:         [20.82, 35.76] → 充分利用色彩范围
Fine-tuned: [0.01, 2.13]   → 全部映射到紫色（看不清细节）
Baseline:   [0.59, 1.76]   → 全部映射到紫色（看不清细节）
```

### 策略 2: 独立归一化（当前使用）✅

```python
# 每个深度图独立归一化到 0-1
gt_colored, gt_min, gt_max = colorize_depth(gt_depth)
ft_colored, ft_min, ft_max = colorize_depth(finetuned_depth)
bl_colored, bl_min, bl_max = colorize_depth(baseline_depth)
```

**优点**:
- ✅ 每个深度图都充分利用色彩范围
- ✅ 可以清楚看到每个模型预测的细节和结构
- ✅ 适合调试阶段，即使深度范围错误也能看清结构
- ✅ 通过标题中的数值范围进行定量对比

**缺点**:
- ⚠️ 色彩不直接对应相同的深度值
- ⚠️ 需要看标题中的数值范围才能定量对比

**示例效果**:
```
GT:         [20.82, 35.76] → 紫(20.82) 到 黄(35.76)
Fine-tuned: [0.01, 2.13]   → 紫(0.01) 到 黄(2.13)  ← 可以看清结构
Baseline:   [0.59, 1.76]   → 紫(0.59) 到 黄(1.76)  ← 可以看清结构
```

## 🎨 色彩映射（Viridis）

```
深度值:  min ────────────────────────────────────── max
色彩:    紫色 ──→ 蓝色 ──→ 绿色 ──→ 黄色
归一化:  0.0 ────────────────────────────────────── 1.0
```

### 独立归一化后的映射

**Ground Truth** (实际深度 20-36 米):
```
20.82m → 紫色 (0.0)
28.29m → 绿色 (0.5)
35.76m → 黄色 (1.0)
```

**Fine-tuned** (预测深度 0-2 米):
```
0.01m → 紫色 (0.0)
1.07m → 绿色 (0.5)
2.13m → 黄色 (1.0)
```

**Baseline** (预测深度 0.6-1.8 米):
```
0.59m → 紫色 (0.0)
1.18m → 绿色 (0.5)
1.76m → 黄色 (1.0)
```

## 📊 如何解读可视化结果

### 1. 结构相似性（视觉对比）
观察三个深度图的**色彩分布模式**:
- 如果 fine-tuned 的色彩模式与 GT 相似 → 结构预测正确 ✅
- 如果 baseline 的色彩模式与 GT 差异大 → 结构预测错误 ❌

### 2. 深度范围（数值对比）
查看标题中的**实际深度范围**:
- GT: [20.82, 35.76] → 真实深度
- Fine-tuned: [0.01, 2.13] → 预测深度（偏小）
- Baseline: [0.59, 1.76] → 预测深度（偏小）

### 3. 定量指标（MAE/RMSE）
结合控制台输出的指标:
```
Fine-tuned MAE: 0.234, RMSE: 0.456  ← 更好
Baseline MAE: 3.567, RMSE: 5.789
```

## ✅ 成功的过拟合应该看到什么

### 视觉上（色彩模式）
- Fine-tuned 的色彩分布与 GT **非常相似**
- Baseline 的色彩分布与 GT **有明显差异**
- 细节、边界、渐变都应该匹配

### 数值上（深度范围）
理想情况（完美过拟合）:
```
GT:         [20.82, 35.76]
Fine-tuned: [20.80, 35.78]  ← 非常接近
Baseline:   [18.50, 38.20]  ← 有差异
```

实际情况（可能需要调试）:
```
GT:         [20.82, 35.76]
Fine-tuned: [0.01, 2.13]    ← 深度尺度错误，需要检查训练
Baseline:   [0.59, 1.76]    ← 深度尺度错误
```

### 指标上（MAE/RMSE）
```
Fine-tuned MAE: < 0.5 米   ← 目标
Baseline MAE: > 2 米
Improvement: > 80%
```

## 🔧 何时使用哪种策略

### 使用独立归一化（当前策略）✅
- ✅ 调试阶段（检查模型是否学到了结构）
- ✅ 深度范围差异很大
- ✅ 需要看清每个模型的细节
- ✅ 单 pair 过拟合测试

### 使用统一范围（可选）
- 如果模型已经收敛，深度范围接近 GT
- 需要直接视觉对比深度值
- 最终展示阶段

## 💡 实现细节

### 独立归一化函数
```python
def colorize_depth(depth_map, cmap='viridis'):
    """每个深度图独立归一化到 0-1"""
    valid_mask = depth_map > 0
    vmin = depth_map[valid_mask].min()
    vmax = depth_map[valid_mask].max()
    
    # 归一化到 0-1
    normalized = (depth_map - vmin) / (vmax - vmin)
    
    # 应用色彩映射
    colored = colormap(normalized)
    
    return colored_rgb, vmin, vmax
```

### 可视化输出
```python
# 显示原始深度范围
axes[1].set_title(f'Ground Truth\nRange: [{gt_min:.2f}, {gt_max:.2f}]')
axes[2].set_title(f'Fine-tuned\nRange: [{ft_min:.2f}, {ft_max:.2f}]')
axes[3].set_title(f'Baseline\nRange: [{bl_min:.2f}, {bl_max:.2f}]')
```

## 📝 总结

**当前策略（独立归一化）的核心思想**:
1. 用**色彩模式**看结构相似性（视觉）
2. 用**数值范围**看深度准确性（定量）
3. 用**MAE/RMSE**看整体误差（指标）

这样可以同时评估模型的**结构预测能力**和**深度估计准确性**。

---

**更新日期**: 2025-10-19  
**策略**: 独立归一化（每个深度图 0-1）  
**目的**: 清楚看到细节 + 数值对比
