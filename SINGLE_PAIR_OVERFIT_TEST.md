# 单 Pair 过拟合测试指南

## 🎯 目的

创建一个只包含 **1 个 pair** 的数据集，用于验证模型能否在单个样本上将 loss 降到 0。这是验证模型实现正确性的重要测试。

---

## 📦 数据集创建

### 1. 运行创建脚本

```bash
python create_single_pair_dataset.py --create_config
```

**参数说明**:
- `--source_root`: 源数据集根目录（默认：`training/dataset_aerialmd/cropped`）
- `--source_seg_root`: 源分割掩码根目录（默认：`training/dataset_aerialmd/cropped_seg`）
- `--source_npz`: 源 NPZ 文件名（默认：`train.npz`）
- `--output_root`: 输出数据集根目录（默认：`training/dataset_aerialmd_single`）
- `--pair_index`: 要提取的 pair 索引（默认：`0`）
- `--create_config`: 是否创建配置文件

### 2. 输出结果

**数据集结构**:
```
training/dataset_aerialmd_single/
├── train.npz                                    # 1 个 pair
├── val.npz                                      # 1 个 pair (相同)
├── 0001/                                        # 场景目录
│   ├── 3775224815_2e30aeddbb_o.jpg.jpg.jpg     # RGB 图像 1
│   ├── 3775224815_2e30aeddbb_o.jpg.jpg.exr     # Depth map 1
│   ├── 3775224815_2e30aeddbb_o.jpg.jpg.npz     # Camera params 1
│   ├── 0001_083.jpeg.jpg                        # RGB 图像 2
│   ├── 0001_083.jpeg.exr                        # Depth map 2
│   └── 0001_083.jpeg.npz                        # Camera params 2
└── seg/                                         # 分割掩码目录
    └── 0001/
        ├── 3775224815_2e30aeddbb_o.jpg.jpg.png # Segmentation mask 1
        └── 0001_083.jpeg.png                    # Segmentation mask 2
```

**配置文件**:
- `training/config/single_pair_test.yaml` - 自动生成的训练配置

---

## 🧪 测试数据集

### 验证数据集是否正确创建

```bash
python test_single_pair_dataset.py
```

**预期输出**:
```
✅ 配置加载成功
✅ Train dataset 创建成功
   - 数据集长度: 1
✅ Loader 创建成功

批次 0:
  - seq_name: ['aerial_megadepth_0001_0']
  - images 形状: torch.Size([1, 2, 3, 476, 518])
  - depths 形状: torch.Size([1, 2, 476, 518])
  - 样本 0 图像 0: 有效深度 213634/246568 (86.6%), 范围 [20.82, 35.27]
  - 样本 0 图像 1: 有效深度 130264/246568 (52.8%), 范围 [41.07, 58.34]
```

---

## 🚀 运行过拟合测试

### 1. 启动训练

```bash
cd training
python launch.py --config single_pair_test
```

### 2. 配置说明

**关键配置** (`training/config/single_pair_test.yaml`):

```yaml
exp_name: single_pair_overfit_test
max_epochs: 100  # 训练 100 个 epoch

data:
  train:
    dataset:
      dataset_configs:
        - ROOT: /path/to/dataset_aerialmd_single
          split_file: train.npz
          segmentation_root: /path/to/dataset_aerialmd_single/seg
          len_train: 1  # 只有 1 个 pair

optim:
  optimizer:
    lr: 1e-4  # 较高的学习率，加快过拟合
    weight_decay: 0.01  # 较小的权重衰减
  
  frozen_module_names: []  # 不冻结任何模块

logging:
  log_freq: 1  # 每个 batch 都记录
  log_visuals: True  # 启用可视化
```

### 3. 监控训练

**TensorBoard**:
```bash
tensorboard --logdir logs/single_pair_test/tensorboard
```

**关键指标**:
- `loss_objective`: 总损失
- `loss_camera`: 相机参数损失
- `loss_conf_depth`: 深度置信度损失
- `loss_reg_depth`: 深度回归损失

---

## 📊 预期结果

### ✅ 成功的过拟合

如果模型实现正确，应该观察到：

1. **Loss 持续下降**
   - 前 10 个 epoch: loss 快速下降
   - 10-50 个 epoch: loss 继续下降
   - 50-100 个 epoch: loss 趋近于 0

2. **各项损失都下降**
   - `loss_camera` → 接近 0
   - `loss_conf_depth` → 接近 0
   - `loss_reg_depth` → 接近 0

3. **训练和验证 loss 一致**
   - 因为使用相同的数据，train 和 val loss 应该几乎相同

### ⚠️ 可能的问题

如果 loss 不下降或下降很慢：

1. **学习率问题**
   - 尝试调整 `lr`: `1e-4`, `5e-5`, `1e-5`

2. **模型冻结问题**
   - 检查 `frozen_module_names` 是否为空
   - 确保所有模块都在训练

3. **数据问题**
   - 检查深度图是否有效
   - 检查分割掩码是否正确应用

4. **损失权重问题**
   - 调整 `loss.camera.weight` 和 `loss.depth.weight`

---

## 🔍 调试技巧

### 1. 检查数据加载

```python
# 在 trainer.py 的 train_epoch 中添加
print(f"Batch keys: {batch.keys()}")
print(f"Images shape: {batch['images'].shape}")
print(f"Depths shape: {batch['depths'].shape}")
print(f"Valid depth ratio: {(batch['depths'] > 0).float().mean()}")
```

### 2. 检查梯度

```python
# 在 trainer.py 的 train_epoch 中添加
for name, param in self.model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm()}")
```

### 3. 检查损失

```python
# 在 trainer.py 的 train_epoch 中添加
print(f"Loss breakdown:")
for key, value in losses.items():
    print(f"  {key}: {value.item():.6f}")
```

### 4. 可视化预测

启用 `log_visuals: True` 后，在 TensorBoard 中查看：
- 输入图像
- 预测深度图
- Ground truth 深度图
- 深度误差图

---

## 📝 实验记录模板

```
实验日期: [填写日期]
配置文件: single_pair_test.yaml

数据集信息:
- Pair 数量: 1
- 场景: 0001
- 图像 1: 3775224815_2e30aeddbb_o.jpg.jpg
- 图像 2: 0001_083.jpeg
- 有效深度比例: 86.6% / 52.8%

训练配置:
- 学习率: 1e-4
- Epochs: 100
- Batch size: 1
- 冻结模块: 无

结果:
Epoch | Train Loss | Val Loss | Camera Loss | Depth Loss
------|-----------|----------|-------------|------------
1     | [填写]    | [填写]   | [填写]      | [填写]
10    | [填写]    | [填写]   | [填写]      | [填写]
50    | [填写]    | [填写]   | [填写]      | [填写]
100   | [填写]    | [填写]   | [填写]      | [填写]

观察:
- Loss 是否下降: [是/否]
- 是否过拟合成功: [是/否]
- 最终 loss 值: [填写]
- 遇到的问题: [填写]

结论:
[填写结论]
```

---

## 🎯 成功标准

### ✅ 模型实现正确

如果满足以下条件，说明模型实现正确：

1. **Loss 能降到接近 0**
   - `loss_objective < 0.01`
   - `loss_camera < 0.001`
   - `loss_depth < 0.01`

2. **训练稳定**
   - 没有 NaN 或 Inf
   - 梯度正常
   - Loss 单调下降

3. **预测准确**
   - 预测深度图与 GT 接近
   - 相机参数预测准确

### ❌ 需要调试

如果出现以下情况，需要检查实现：

1. **Loss 不下降**
   - 检查数据加载
   - 检查损失计算
   - 检查梯度流

2. **Loss 震荡**
   - 降低学习率
   - 检查数据归一化
   - 检查损失权重

3. **Loss 下降很慢**
   - 增加学习率
   - 检查模型是否被冻结
   - 检查优化器配置

---

## 📚 相关文档

- `create_single_pair_dataset.py` - 数据集创建脚本
- `test_single_pair_dataset.py` - 数据集测试脚本
- `training/config/single_pair_test.yaml` - 训练配置
- `EXACT_TRAINING_VERIFICATION.md` - 完整数据集验证

---

## 🔧 故障排除

### 问题 1: 数据集创建失败

```bash
# 检查源数据集
ls training/dataset_aerialmd/cropped/
ls training/dataset_aerialmd/cropped_seg/

# 重新创建
python create_single_pair_dataset.py --create_config
```

### 问题 2: 配置文件找不到

```bash
# 检查配置文件
ls training/config/single_pair_test.yaml

# 手动创建
python create_single_pair_dataset.py --create_config
```

### 问题 3: 训练启动失败

```bash
# 检查分布式初始化
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# 使用单 GPU
CUDA_VISIBLE_DEVICES=0 python launch.py --config single_pair_test
```

---

**创建时间**: 2025-10-19  
**用途**: 验证模型能否在单个样本上过拟合  
**预期结果**: Loss 降到接近 0
