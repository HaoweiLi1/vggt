# VGGT AerialMegaDepth 数据加载测试指南

本文档提供了完整的测试流程，用于验证 VGGT 中 AerialMegaDepth 数据集的加载和处理是否正确。

---

## 📋 测试文件说明

### 1. `test_vggt_aerial_dataloader.py`
**完整功能测试脚本**

测试内容：
- ✅ 数据加载器初始化
- ✅ RGB 图像加载
- ✅ Depth Map 加载和处理
- ✅ Camera Parameters 加载和坐标系转换
- ✅ Segmentation Mask 应用（天空移除）
- ✅ 批次数据生成
- ✅ 深度过滤策略
- ✅ 多批次稳定性测试

### 2. `visualize_aerial_data.py`
**可视化测试脚本**

生成可视化图像：
- RGB 图像
- 原始深度图
- 分割掩码
- 天空区域标注
- 天空移除后的深度图
- 完整处理后的深度图
- 批次数据可视化

### 3. `test_config_aerial.yaml`
**测试配置文件**

用于快速测试数据加载器的配置文件。

### 4. `test_segmentation_mask.py`
**分割掩码单独测试**

专门测试分割掩码的应用效果。

---

## 🚀 运行测试

### 方法 1: 完整功能测试（推荐）

```bash
cd /home/haowei/Documents/vggt
python test_vggt_aerial_dataloader.py
```

**预期输出：**
```
============================================================
VGGT AerialMegaDepth 数据加载器完整测试
============================================================

测试 1: 数据加载器初始化
============================================================
✅ 数据集初始化成功
   - 场景数量: 5
   - 有效场景: ['0000', '0001', '0002', '0003', '0015']
   - 配对数量: XXX
   - 数据集长度: XXX
   - 分割掩码路径: /home/haowei/Documents/vggt/training/dataset_aerialmd/cropped_seg
   - 天空移除: True

测试 2: 单张图像加载
============================================================
✅ 图像加载成功
   - RGB 尺寸: (518, 518, 3)
   - Depth 尺寸: (518, 518)
   ...

[更多测试输出]

============================================================
测试总结
============================================================
   初始化: ✅ 通过
   单张图像加载: ✅ 通过
   分割掩码应用: ✅ 通过
   坐标系转换: ✅ 通过
   批次生成: ✅ 通过
   深度过滤: ✅ 通过
   多批次稳定性: ✅ 通过

   总计: 7/7 测试通过

🎉 所有测试通过！VGGT 正确使用了 AerialMegaDepth 数据集！
============================================================
```

---

### 方法 2: 可视化测试

```bash
cd /home/haowei/Documents/vggt
python visualize_aerial_data.py
```

**输出：**
- 在 `test_visualizations/` 目录下生成可视化图像
- 每个样本生成一张 6 宫格图像，展示完整的处理流程
- 生成批次数据的可视化

**查看结果：**
```bash
ls test_visualizations/
# sample_0_0001_0001_001.jpeg.png
# sample_1_0001_0001_002.jpeg.png
# batch_aerial_megadepth_0001_0.png
```

---

### 方法 3: 分割掩码单独测试

```bash
cd /home/haowei/Documents/vggt
python test_segmentation_mask.py
```

**预期输出：**
```
============================================================
Testing Segmentation Mask Application
============================================================

1. Loading depth map from: ...
   ✓ Depth map shape: (518, 518)
   ✓ Depth range: [0.00, 444.04]
   ✓ Valid depth pixels: 183189 / 268324

2. Loading segmentation mask from: ...
   ✓ Segmentation mask shape: (518, 518)
   ✓ Unique labels: [0, 1, 2, 12, 17, 43, 132]
   ✓ Sky pixels (label=2): 77586 / 268324

3. Applying segmentation mask to remove sky...
   ✓ Sky pixels with valid depth (before): 5512
   ✓ Sky pixels with valid depth (after): 0

✅ SUCCESS: Segmentation mask correctly removes sky regions!
```

---

## 🔍 测试检查清单

运行测试后，请确认以下内容：

### ✅ 数据加载
- [ ] 数据集成功初始化
- [ ] 场景目录被正确识别
- [ ] NPZ 文件被正确加载
- [ ] 配对数量合理

### ✅ RGB 图像
- [ ] 图像尺寸正确 (518x518x3)
- [ ] 图像值范围正确 (0-255)
- [ ] 图像内容清晰可见

### ✅ Depth Map
- [ ] 深度图尺寸正确 (518x518)
- [ ] 深度值范围合理 (0-2000)
- [ ] 有效深度像素比例合理 (>50%)

### ✅ Segmentation Mask
- [ ] 掩码文件存在
- [ ] 掩码尺寸与图像匹配
- [ ] 天空标签 (2) 被正确识别
- [ ] 天空区域深度值被设为 0

### ✅ Camera Parameters
- [ ] 内参矩阵格式正确 (3x3)
- [ ] 外参矩阵格式正确 (3x4)
- [ ] cam2world → world2cam 转换正确
- [ ] 逆矩阵验证通过

### ✅ 批次生成
- [ ] 批次包含所有必需的键
- [ ] 每个批次至少有 2 张图像
- [ ] 点云数据正确生成
- [ ] 掩码数据正确生成

### ✅ 深度过滤
- [ ] 硬阈值过滤工作正常
- [ ] 百分位数过滤工作正常
- [ ] 过滤后仍有足够的有效深度

---

## 🐛 常见问题排查

### 问题 1: 找不到模块
```
ModuleNotFoundError: No module named 'data'
```

**解决方案：**
```bash
cd /home/haowei/Documents/vggt
export PYTHONPATH=$PYTHONPATH:$(pwd)/training
python test_vggt_aerial_dataloader.py
```

### 问题 2: OpenEXR 未启用
```
cv2.error: OpenCV(4.11.0) ... OpenEXR codec is disabled
```

**解决方案：**
脚本已自动设置 `os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"`，如果仍有问题，请检查 OpenCV 安装。

### 问题 3: 分割掩码文件不存在
```
⚠️ 分割掩码不存在: ...
```

**解决方案：**
```bash
# 检查路径
ls /home/haowei/Documents/vggt/training/dataset_aerialmd/cropped_seg/

# 如果缺少，运行裁剪脚本
python process_crop.py \
    --batch_input training/dataset_aerialmd/processed_seg \
    --batch_output training/dataset_aerialmd/cropped_seg \
    --file_types png
```

### 问题 4: 数据集为空
```
RuntimeError: No valid scene directories found
```

**解决方案：**
```bash
# 检查数据目录
ls /home/haowei/Documents/vggt/training/dataset_aerialmd/cropped/

# 确保有场景目录 (0000, 0001, 0002, 0003, 0015)
```

---

## 📊 预期测试结果

### 成功标准

所有 7 个测试都应该通过：

1. ✅ **初始化** - 数据集成功加载，场景和配对数量正确
2. ✅ **单张图像加载** - RGB、深度、相机参数正确加载
3. ✅ **分割掩码应用** - 天空区域深度值被正确移除
4. ✅ **坐标系转换** - cam2world 正确转换为 world2cam
5. ✅ **批次生成** - 批次数据完整且格式正确
6. ✅ **深度过滤** - 硬阈值和百分位数过滤正常工作
7. ✅ **多批次稳定性** - 连续多个批次都能正常加载

### 关键指标

- **有效深度像素比例**: 应该在 50-80% 之间
- **天空覆盖率**: 通常在 20-40% 之间
- **天空移除效果**: 天空区域深度值应该全部为 0
- **批次成功率**: 应该达到 100%

---

## 📝 测试报告模板

运行测试后，可以使用以下模板记录结果：

```
VGGT AerialMegaDepth 数据加载测试报告
日期: [填写日期]
测试人: [填写姓名]

环境信息:
- Python 版本: [填写]
- PyTorch 版本: [填写]
- OpenCV 版本: [填写]

数据集信息:
- 场景数量: [填写]
- 配对数量: [填写]
- 数据路径: /home/haowei/Documents/vggt/training/dataset_aerialmd/cropped

测试结果:
[ ] 初始化测试
[ ] 单张图像加载测试
[ ] 分割掩码应用测试
[ ] 坐标系转换测试
[ ] 批次生成测试
[ ] 深度过滤测试
[ ] 多批次稳定性测试

总体评估:
[ ] 通过 - 所有测试通过，数据加载正确
[ ] 部分通过 - 部分测试失败，需要修复
[ ] 失败 - 主要测试失败，需要重新配置

备注:
[填写任何额外的观察或问题]
```

---

## 🎯 下一步

测试通过后，可以：

1. **开始训练**
   ```bash
   cd training
   python train.py --config config/default.yaml
   ```

2. **监控训练**
   - 检查 TensorBoard 日志
   - 验证损失函数下降
   - 检查深度预测质量

3. **调整参数**
   - 如果深度损失异常，调整 `max_depth` 或 `depth_percentile`
   - 如果需要更严格的天空过滤，可以调整分割阈值

---

## 📚 相关文档

- `dataset_usage_comparison.md` - 详细的实现对比
- `comparison_summary.md` - 快速对比总结
- `segmentation_check_report.md` - 分割掩码验证报告

---

## ✅ 结论

如果所有测试通过，说明：

1. ✅ VGGT 正确加载了 AerialMegaDepth 数据
2. ✅ RGB、Depth、Camera Parameters 处理正确
3. ✅ Segmentation Mask 正确应用（天空移除）
4. ✅ 坐标系转换正确（cam2world → world2cam）
5. ✅ 数据批次生成稳定可靠

**可以放心使用 VGGT 训练 AerialMegaDepth 数据集！** 🚀
