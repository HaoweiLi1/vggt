# 快速参考卡片

## 🚀 快速开始

```bash
# 运行对比
./run_comparison.sh

# 查看结果
ls -lh model_comparison/
```

## 📊 可视化布局

```
┌─────────────────────────────────────────────────────────────────┐
│              Model Comparison: 0001/image_name                  │
├──────────┬──────────────┬──────────────┬──────────────────────┤
│   RGB    │ Ground Truth │  Fine-tuned  │      Baseline        │
│  Image   │    Depth     │    Model     │       Model          │
│          │              │              │                      │
│          │ Range:       │ Range:       │ Range:               │
│          │ [20.8, 35.8] │ [0.01, 2.13] │ [0.59, 1.76]         │
│          │              │              │                      │
│          │ Valid:       │ MAE: 46.427  │ MAE: 46.458          │
│          │ 268324/      │ RMSE: 46.938 │ RMSE: 46.980         │
│          │ 268324       │ Abs Rel: 1.6 │ Abs Rel: 1.6         │
│          │ (100%)       │              │                      │
└──────────┴──────────────┴──────────────┴──────────────────────┘
                    Improvement:
                    MAE: -0.1%
                    RMSE: -0.1%
```

## 🎯 成功标准速查

### ✅ 过拟合成功
| 指标 | Fine-tuned | Baseline | 改进 |
|------|-----------|----------|------|
| MAE | < 0.5 米 | > 2 米 | > 80% |
| RMSE | < 1.0 米 | > 3 米 | > 80% |
| Abs Rel | < 0.1 | > 0.2 | > 50% |
| 深度范围 | 接近 GT | 偏离 GT | - |
| 色彩模式 | 与 GT 相似 | 与 GT 不同 | - |

### ❌ 需要调试
| 现象 | 可能原因 | 解决方向 |
|------|---------|---------|
| 改进 < 20% | 训练未收敛 | 增加训练轮数 |
| 深度范围偏小 | 尺度错误 | 检查深度归一化 |
| 色彩模式不同 | 结构错误 | 检查模型架构 |
| MAE 很大 | 整体偏移 | 检查损失权重 |

## 📈 指标速查

### MAE (Mean Absolute Error)
- **含义**: 平均绝对误差
- **单位**: 米
- **目标**: < 0.5 米
- **公式**: `mean(|pred - gt|)`

### RMSE (Root Mean Square Error)
- **含义**: 均方根误差
- **单位**: 米
- **目标**: < 1.0 米
- **公式**: `sqrt(mean((pred - gt)²))`

### Abs Rel (Absolute Relative Error)
- **含义**: 绝对相对误差
- **单位**: 无量纲（比例）
- **目标**: < 0.1 (10%)
- **公式**: `mean(|pred - gt| / gt)`

## 🎨 色彩映射

```
Viridis 色彩映射（每个深度图独立归一化到 0-1）:

深度值:  min ──────────────────────────────── max
色彩:    紫色 ──→ 蓝色 ──→ 绿色 ──→ 黄色
归一化:  0.0 ───────────────────────────────── 1.0
```

**注意**: 不同深度图的颜色不直接对应相同的深度值！  
需要看标题中的 `Range: [min, max]` 来判断实际深度。

## 🔍 快速诊断

### 1. 看色彩模式
```python
if fine_tuned_colors_similar_to_gt:
    print("✅ 结构预测正确")
else:
    print("❌ 结构预测错误，检查模型架构")
```

### 2. 看深度范围
```python
if fine_tuned_range_close_to_gt:
    print("✅ 尺度预测正确")
else:
    print("❌ 尺度预测错误，检查深度归一化")
```

### 3. 看指标数值
```python
if mae < 0.5 and improvement > 80:
    print("✅ 过拟合成功！")
else:
    print("❌ 需要继续训练或调试")
```

## 📁 文件说明

| 文件 | 用途 |
|------|------|
| `compare_models.py` | 主对比脚本 |
| `run_comparison.sh` | 便捷运行脚本 |
| `COMPARISON_USAGE.md` | 详细使用指南 |
| `METRICS_EXPLANATION.md` | 指标详细说明 |
| `VISUALIZATION_STRATEGY.md` | 可视化策略说明 |
| `MEMORY_OPTIMIZATION_SUMMARY.md` | 显存优化说明 |
| `QUICK_REFERENCE.md` | 本文档（快速参考）|

## 🐛 常见问题

### Q1: CUDA out of memory
**A**: 脚本已优化为分阶段加载模型，应该不会出现此问题。如果仍然出现，检查是否有其他程序占用 GPU。

### Q2: 深度图全是紫色
**A**: 这是正常的！使用独立归一化后，每个深度图都会充分利用色彩范围。看标题中的实际深度范围来判断。

### Q3: 改进为负数
**A**: 说明 fine-tuned 模型比 baseline 更差，需要检查训练过程。

### Q4: 指标显示 "No metrics"
**A**: 没有 Ground Truth 深度图，无法计算指标。检查数据目录。

## 💡 提示

1. **第一次运行**: 先运行 `python test_model_loading.py` 测试模型加载
2. **查看进度**: 脚本会显示 3 个阶段的进度（Fine-tuned → Baseline → 可视化）
3. **显存监控**: 运行 `watch -n 1 nvidia-smi` 监控显存使用
4. **结果分析**: 结合可视化图和控制台输出进行综合分析

---

**快速参考版本**: v1.0  
**更新日期**: 2025-10-19
