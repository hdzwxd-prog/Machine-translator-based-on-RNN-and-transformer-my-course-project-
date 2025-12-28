# 快速开始指南

本指南帮助你快速上手神经机器翻译项目。

## 1. 环境准备

### 1.1 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate
```

### 1.2 安装依赖

```bash
pip install -r requirements.txt
```

如果使用CPU训练（推荐首次测试）：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy jieba nltk pyyaml tqdm sacrebleu
```

如果使用GPU训练（需要CUDA）：
```bash
pip install torch
pip install numpy jieba nltk pyyaml tqdm sacrebleu
```

### 1.3 下载NLTK数据（必需）

```bash
python setup_nltk.py
```

### 1.4 验证安装

```bash
python check_project.py
```

应该看到所有检查通过的信息。

## 2. 配置修改（可选）

编辑 `config/config.yaml`，根据你的需求调整：

### 快速测试配置（小规模）
```yaml
data:
  train_file: "train_10k.jsonl"  # 使用10k数据集
  max_length: 50                  # 减小最大长度

model:
  encoder:
    hidden_dim: 256               # 减小隐藏层维度
  decoder:
    hidden_dim: 256

training:
  batch_size: 32                  # 减小批次大小
  num_epochs: 5                   # 减少训练轮数
```

### 完整训练配置（标准）
```yaml
data:
  train_file: "train_100k.jsonl"  # 使用100k数据集
  max_length: 100

model:
  encoder:
    hidden_dim: 512
  decoder:
    hidden_dim: 512

training:
  batch_size: 64
  num_epochs: 30
```

## 3. 训练模型

### 3.1 基础训练

```bash
python train.py
```

训练过程会显示：
- 数据加载进度
- 词汇表大小
- 模型参数数量
- 每个epoch的训练/验证损失
- 自动保存最佳模型

### 3.2 训练输出

训练会生成以下文件：
- `vocabs/src_vocab.pkl` - 源语言词汇表
- `vocabs/tgt_vocab.pkl` - 目标语言词汇表
- `checkpoints/best_model.pt` - 最佳模型
- `checkpoints/checkpoint_epoch_*.pt` - 定期检查点

### 3.3 训练时间估计

在CPU上：
- 10k数据集：约30分钟-1小时
- 100k数据集：约3-5小时

在GPU上：
- 10k数据集：约5-10分钟
- 100k数据集：约30-60分钟

## 4. 评估模型

### 4.1 贪心解码

```bash
python evaluate.py --strategy greedy --output results_greedy.txt
```

### 4.2 束搜索解码

```bash
python evaluate.py --strategy beam_search --beam_size 5 --output results_beam5.txt
```

### 4.3 评估输出

评估脚本会：
1. 加载模型和词汇表
2. 翻译测试集
3. 计算BLEU分数
4. 保存结果到指定文件
5. 打印翻译示例

示例输出：
```
BLEU Score: 25.34

样本 1:
源句子: 这 是 一个 测试 句子
参考翻译: this is a test sentence
模型翻译: this is a test sentence
```

## 5. 对比实验

### 5.1 手动对比实验

#### 实验A：对比注意力机制

1. 修改 `config/config.yaml`：
```yaml
model:
  attention:
    type: "dot"  # 改为 general 或 additive
```

2. 训练：
```bash
python train.py
mv checkpoints/best_model.pt checkpoints/best_model_dot.pt
```

3. 对其他注意力类型重复步骤1-2

4. 评估对比：
```bash
python evaluate.py --checkpoint checkpoints/best_model_dot.pt
python evaluate.py --checkpoint checkpoints/best_model_general.pt
python evaluate.py --checkpoint checkpoints/best_model_additive.pt
```

#### 实验B：对比Teacher Forcing

1. 修改配置：
```yaml
training:
  teacher_forcing_ratio: 1.0  # 改为 0.5 或 0.0
```

2. 训练并评估多个比例

#### 实验C：对比解码策略

使用同一个模型，对比不同解码策略：
```bash
python evaluate.py --strategy greedy
python evaluate.py --strategy beam_search --beam_size 3
python evaluate.py --strategy beam_search --beam_size 5
python evaluate.py --strategy beam_search --beam_size 10
```

### 5.2 自动批量实验（高级）

如果你想自动运行所有对比实验：

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

这会自动运行所有实验组合并保存结果到 `experiments/` 目录。

⚠️ 注意：这会花费很长时间（几个小时到一天）。

## 6. 分析结果

### 6.1 查看BLEU分数

```bash
grep "BLEU Score" experiments/results_*.txt
```

### 6.2 对比训练曲线

如果安装了tensorboard（可选）：
```bash
pip install tensorboard
tensorboard --logdir=runs
```

### 6.3 分析翻译质量

查看各个结果文件中的翻译示例，对比：
- 流畅度
- 准确度
- 长句处理能力
- 罕见词处理

## 7. 常见问题

详细问题解答请参考 `README.md` 的"常见问题"部分。

### 快速排查

1. **NLTK数据未找到**：运行 `python setup_nltk.py`
2. **分布式训练卡住**：运行 `./debug_distributed.sh` 检查环境
3. **FP16溢出错误**：代码已自动修复，如仍有问题检查attention.py
4. **内存不足**：减小batch_size和hidden_dim
5. **训练太慢**：使用GPU，启用分布式训练和混合精度

## 8. 下一步

完成基础训练和评估后：

1. **撰写实验报告**：
   - 记录各配置的BLEU分数
   - 分析不同策略的优劣
   - 展示翻译示例

2. **扩展项目**（可选）：
   - 实现Transformer模型
   - 添加预训练词向量
   - 实现更多评估指标
   - 添加可视化（注意力热图）

3. **优化模型**：
   - 超参数搜索
   - 集成学习
   - 后处理优化

## 9. 获取帮助

- 查看 `README.md` 了解详细功能说明
- 查看代码注释了解实现细节
- 查看配置文件了解可调参数

祝你实验顺利！🚀

