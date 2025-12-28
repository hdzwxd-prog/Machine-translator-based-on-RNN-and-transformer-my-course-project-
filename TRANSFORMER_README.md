# Transformer模型使用指南

本文档说明如何使用项目中新增的Transformer模型模块。

## 概述

项目现在支持三种模型类型：
1. **RNN模型**：原有的RNN-based Seq2Seq模型
2. **Transformer模型**：从零构建的Transformer架构
3. **T5模型**：基于预训练T5模型的微调

## 快速开始

### 1. Transformer模型训练

使用Transformer配置文件进行训练：

```bash
python train.py --config config/config_transformer.yaml
```

### 2. 架构消融实验

#### 对比位置编码方案

**绝对位置编码（Absolute Positional Encoding）**：
```bash
python train.py --config config/config_transformer.yaml
```

**相对位置编码（Relative Positional Encoding）**：
```bash
python train.py --config config/config_transformer_relative.yaml
```

#### 对比归一化方法

**LayerNorm**：
```bash
python train.py --config config/config_transformer.yaml
```

**RMSNorm**：
```bash
python train.py --config config/config_transformer_rmsnorm.yaml
```

### 3. T5预训练模型微调

```bash
python train.py --config config/config_t5.yaml
```

**注意**：T5模型需要安装transformers库：
```bash
pip install transformers
```

### 4. transformer 模型评估

#### 使用贪心解码
```bash
python evaluate.py \
    --config config/config_transformer.yaml \
    --checkpoint checkpoints_transformer/best_model.pt \
    --strategy greedy \
    --output results_transformer_greedy.txt
```
#### 使用束搜索解码
```bash
python evaluate.py \
    --config config/config_transformer.yaml \
    --checkpoint checkpoints_transformer/best_model.pt \
    --strategy beam_search \
    --beam_size 5 \
    --output results_transformer_beam5.txt
```
## 配置文件说明

### Transformer配置（config_transformer.yaml）

主要配置项：
- `model.type`: `"transformer"` - 指定使用Transformer模型
- `model.pos_encoding_type`: `"absolute"` 或 `"relative"` - 位置编码类型
- `model.norm_type`: `"layernorm"` 或 `"rmsnorm"` - 归一化方法
- `model.encoder.d_model`: 模型维度（默认512）
- `model.encoder.num_heads`: 注意力头数（默认8）
- `model.encoder.num_layers`: 编码器层数（默认6）
- `model.encoder.d_ff`: 前馈网络隐藏层维度（默认2048）

### T5配置（config_t5.yaml）

主要配置项：
- `model.type`: `"t5"` - 指定使用T5模型
- `model.model_name`: `"t5-small"`, `"t5-base"`, 或 `"t5-large"` - T5模型大小

## 模型架构

### Transformer模型组件

1. **位置编码** (`src/models/transformer/positional_encoding.py`)
   - `AbsolutePositionalEncoding`: 绝对位置编码（sin/cos）
   - `RelativePositionalEncoding`: 相对位置编码（可学习）

2. **归一化** (`src/models/transformer/normalization.py`)
   - `LayerNorm`: 层归一化
   - `RMSNorm`: 均方根归一化

3. **注意力机制** (`src/models/transformer/attention.py`)
   - `MultiHeadAttention`: 多头注意力

4. **编码器/解码器** (`src/models/transformer/encoder.py`, `decoder.py`)
   - `TransformerEncoder`: Transformer编码器
   - `TransformerDecoder`: Transformer解码器

5. **Seq2Seq模型** (`src/models/transformer/seq2seq.py`)
   - `TransformerSeq2Seq`: 完整的Transformer Seq2Seq模型

## 超参数敏感性分析

可以通过修改配置文件中的以下参数进行超参数敏感性分析：

1. **批次大小（Batch Size）**
   ```yaml
   training:
     batch_size: 16  # 尝试 8, 16, 32, 64
   ```

2. **学习率（Learning Rate）**
   ```yaml
   training:
     learning_rate: 0.0001  # 尝试 0.00001, 0.0001, 0.001
   ```

3. **模型规模**
   ```yaml
   model:
     encoder:
       d_model: 512  # 尝试 256, 512, 1024
       num_layers: 6  # 尝试 3, 6, 12
       num_heads: 8  # 尝试 4, 8, 16
   ```

## 内存优化建议

由于当前GPU正在训练RNN模型，Transformer训练时需要注意内存使用：

1. **减小batch_size**：配置文件已设置为32，可根据实际情况调整
2. **启用混合精度训练**：`use_amp: true`（已启用）
3. **禁用分布式训练**：`use_ddp: false`（避免与RNN训练冲突）
4. **使用较小的模型**：d_model=512, num_layers=6（默认配置）

## 测试验证

运行测试脚本验证Transformer模型：

```bash
python test_transformer.py
```

该脚本会：
- 检查GPU内存状态
- 使用较小的batch_size进行测试训练
- 只训练2个epoch进行快速验证

## 与RNN模型的对比

| 特性 | RNN模型 | Transformer模型 |
|------|---------|----------------|
| 配置文件 | `config/config.yaml` | `config/config_transformer.yaml` |
| Checkpoint目录 | `checkpoints/` | `checkpoints_transformer/` |
| 位置编码 | 无（RNN隐式处理） | 绝对/相对位置编码 |
| 注意力机制 | 单头注意力 | 多头自注意力 |
| 并行化 | 序列化处理 | 完全并行化 |
| 训练速度 | 较慢 | 较快（并行） |
| 内存占用 | 较低 | 较高 |

## 常见问题

### 1. 内存不足错误

**解决方案**：
- 减小batch_size
- 减小d_model或num_layers
- 启用混合精度训练（use_amp: true）

### 2. T5模型下载失败

**解决方案**：
- 检查网络连接
- 使用镜像源或手动下载模型
- 使用较小的模型（t5-small）

### 3. 训练速度慢

**解决方案**：
- 确保使用GPU（device: "cuda"）
- 启用混合精度训练
- 增加num_workers（数据加载）

## 实验建议

1. **架构消融实验**：
   - 分别训练绝对位置编码和相对位置编码版本
   - 分别训练LayerNorm和RMSNorm版本
   - 对比BLEU分数和训练时间

2. **超参数敏感性分析**：
   - 固定其他参数，只改变batch_size，记录性能
   - 固定其他参数，只改变learning_rate，记录性能
   - 固定其他参数，只改变模型规模，记录性能

3. **预训练vs从零训练**：
   - 训练T5微调模型
   - 训练Transformer从零训练模型
   - 对比两者的BLEU分数和训练时间

## 文件结构

```
src/models/transformer/
├── __init__.py
├── attention.py          # 多头注意力
├── decoder.py           # Transformer解码器
├── encoder.py           # Transformer编码器
├── feedforward.py       # 前馈网络
├── normalization.py     # 归一化层（LayerNorm, RMSNorm）
├── positional_encoding.py  # 位置编码（绝对/相对）
├── seq2seq.py          # Transformer Seq2Seq模型
└── t5_finetune.py     # T5微调模型

config/
├── config_transformer.yaml          # Transformer基础配置
├── config_transformer_relative.yaml # 相对位置编码配置
├── config_transformer_rmsnorm.yaml # RMSNorm配置
└── config_t5.yaml                  # T5微调配置
```

## 下一步

完成基础训练后，可以：
1. 进行更详细的超参数搜索
2. 实现更多位置编码方案（如旋转位置编码RoPE）
3. 添加更多评估指标（如METEOR, ROUGE）
4. 实现模型集成（Ensemble）
5. 添加注意力可视化

