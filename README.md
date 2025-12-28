# 神经机器翻译项目 (Neural Machine Translation)

基于RNN的中英文神经机器翻译系统，支持多种注意力机制、训练策略和解码策略。

## 项目结构

```
NLP_project/
├── config/
│   └── config.yaml              # 配置文件
├── src/
│   ├── data/                    # 数据处理模块
│   │   ├── preprocessor.py      # 数据预处理（清洗、分词）
│   │   ├── vocab.py             # 词汇表构建
│   │   └── dataset.py           # PyTorch数据集
│   ├── models/
│   │   └── rnn/                 # RNN模型
│   │       ├── encoder.py       # 编码器（LSTM/GRU）
│   │       ├── decoder.py       # 解码器（带注意力）
│   │       ├── attention.py     # 注意力机制
│   │       └── seq2seq.py       # Seq2Seq模型
│   ├── training/
│   │   └── trainer.py           # 训练器
│   ├── decoding/
│   │   └── decoder_strategy.py  # 解码策略（贪心/束搜索）
│   └── utils/
│       └── metrics.py           # 评估指标（BLEU）
├── train.py                     # 训练脚本
├── evaluate.py                  # 评估脚本
├── requirements.txt             # 依赖包
└── README.md                    # 项目说明
```

## 功能特点

### 1. 数据预处理
- **数据清洗**：移除非法字符，标准化标点符号
- **分词处理**：
  - 中文：使用jieba分词
  - 英文：使用NLTK分词
- **词汇表构建**：支持低频词过滤，限制词汇表大小
- **数据过滤**：过滤过长句子和长度比例异常的句对

### 2. 模型架构
- **编码器**：双层单向LSTM/GRU
- **解码器**：双层单向LSTM/GRU，带注意力机制
- **注意力机制**：
  - 点积注意力 (Dot-Product Attention)
  - 乘法注意力 (General/Multiplicative Attention)
  - 加法注意力 (Additive/Bahdanau Attention)

### 3. 训练策略
- **Teacher Forcing**：使用真实目标词作为解码器输入
- **Free Running**：使用模型预测词作为解码器输入
- 支持可调节的Teacher Forcing比例

### 4. 解码策略
- **贪心解码 (Greedy Decoding)**：每步选择概率最大的词
- **束搜索解码 (Beam Search Decoding)**：保留多个候选序列，支持长度惩罚

### 5. 评估指标
- BLEU分数

## 安装依赖

```bash
pip install -r requirements.txt
```

如果使用CPU训练，可以安装CPU版本的PyTorch：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 使用说明

### 1. 配置

编辑 `config/config.yaml` 文件，配置数据路径、模型参数、训练参数等。

主要配置项：
- `data.data_dir`: 数据集目录
- `model.cell_type`: RNN类型（lstm或gru）
- `model.attention.type`: 注意力类型（dot, general, additive）
- `training.teacher_forcing_ratio`: Teacher Forcing比例
- `decoding.strategy`: 解码策略（greedy或beam_search）

### 2. 训练

```bash
python train.py
```

训练过程中会：
- 自动处理和加载数据
- 构建词汇表并保存到 `vocabs/` 目录
- 训练模型并保存检查点到 `checkpoints/` 目录
- 定期在验证集上评估
- 保存最佳模型为 `checkpoints/best_model.pt`

### 3. 评估

使用贪心解码：
```bash
python evaluate.py --strategy greedy
```

使用束搜索解码：
```bash
python evaluate.py --strategy beam_search --beam_size 5
```

评估结果会保存到 `results.txt`。

### 4. 对比实验

#### 对比注意力机制

修改 `config/config.yaml` 中的 `model.attention.type`：
- `dot`: 点积注意力
- `general`: 乘法注意力
- `additive`: 加法注意力

分别训练和评估，对比性能。

#### 对比训练策略

修改 `config/config.yaml` 中的 `training.teacher_forcing_ratio`：
- `1.0`: 完全Teacher Forcing
- `0.5`: 50% Teacher Forcing
- `0.0`: 完全Free Running

#### 对比解码策略

评估时使用不同的解码策略：
```bash
# 贪心解码
python evaluate.py --strategy greedy

# 束搜索解码（不同束大小）
python evaluate.py --strategy beam_search --beam_size 3
python evaluate.py --strategy beam_search --beam_size 5
python evaluate.py --strategy beam_search --beam_size 10
```

## 实验建议

### 基础实验
1. 使用默认配置训练基础模型
2. 在测试集上评估并记录BLEU分数

### 对比实验

#### 实验1：注意力机制对比
- 保持其他参数不变
- 分别训练dot、general、additive三种注意力机制
- 对比训练速度、收敛性和最终性能

#### 实验2：训练策略对比
- 对比teacher_forcing_ratio = 1.0, 0.5, 0.0
- 观察训练曲线和验证性能

#### 实验3：解码策略对比
- 使用同一个训练好的模型
- 对比greedy和不同beam_size的beam_search
- 分析翻译质量和解码速度

### 分析建议
- 记录训练时间、收敛速度
- 对比不同配置的BLEU分数
- 分析翻译示例，观察不同策略的差异
- 绘制训练/验证损失曲线

## GPU优化配置（双A6000）

针对双A6000 GPU（48GB显存）的优化配置：

### 已实现的优化
- **模型容量提升**：embed_dim=512, hidden_dim=512（可根据显存调整）
- **分布式训练（DDP）**：支持双GPU数据并行，自动同步梯度
- **混合精度训练（AMP）**：FP16前向传播，FP32梯度计算，节省显存并加速
- **数据加载优化**：num_workers=8, pin_memory=True

### 使用方法
```bash
# 分布式训练（双GPU）
./train_ddp.sh

# 单GPU训练
./train_single.sh
```

### 配置说明
编辑 `config/config.yaml`：
- `training.use_ddp: true` - 启用分布式训练
- `training.use_amp: true` - 启用混合精度训练
- `training.batch_size: 128` - 单卡batch_size（分布式总batch_size=256）
- `training.num_workers: 8` - 数据加载进程数

### 性能预期
- 训练速度：6-8倍提升（相比单GPU小模型）
- 显存使用：~20-25GB/GPU（A6000 48GB足够）
- 单epoch时间：~1小时（100k数据集，双GPU+AMP）

## 扩展方向

本项目架构设计便于扩展，可以添加：

### Transformer模型
在 `src/models/transformer/` 目录下实现Transformer编码器-解码器：
- Multi-Head Attention
- Position Encoding
- Feed-Forward Networks

### 更多功能
- 预训练词向量（Word2Vec, GloVe）
- Learning rate scheduling
- Beam search多样性优化
- 更多评估指标（METEOR, ROUGE等）

## 常见问题

### 内存不足
- 减小batch_size
- 减小max_length
- 减小模型hidden_dim
- 使用梯度累积

### 训练太慢
- 使用GPU（修改config中的device为cuda）
- 减小数据集大小
- 减小模型层数

### 翻译质量差
- 增加训练数据
- 调整teacher_forcing_ratio
- 尝试不同的注意力机制
- 增加模型容量
- 训练更多epoch

## 参考文献

- Bahdanau et al. (2014). Neural Machine Translation by Jointly Learning to Align and Translate
- Luong et al. (2015). Effective Approaches to Attention-based Neural Machine Translation
- Sutskever et al. (2014). Sequence to Sequence Learning with Neural Networks

## 作者

NLP课程项目

## 许可证

MIT License

