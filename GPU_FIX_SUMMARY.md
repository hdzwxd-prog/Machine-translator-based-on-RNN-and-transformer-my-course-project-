# GPU设备选择问题修复总结

## 问题描述

运行 `python train.py --config config/config_transformer.yaml` 时，GPU1占用一直接近0%，怀疑实际在使用GPU0。

## 问题原因

1. **配置文件问题**：`config_transformer.yaml` 中 `device: "cuda"` 会默认使用GPU0
2. **缺少设备设置**：虽然创建了 `torch.device('cuda:1')`，但没有调用 `torch.cuda.set_device(1)` 来设置当前CUDA设备
3. **设备验证不足**：没有验证模型和数据是否真的在指定的GPU上

## 修复方案

### 1. 修改配置文件

在 `config/config_transformer.yaml` 中：
```yaml
# 设备配置
device: "cuda:1"  # 明确指定使用GPU 1
gpu_id: 1  # GPU ID（备用选项）
```

### 2. 修改设备选择逻辑

在 `train.py` 的 `main()` 函数中：
- 检查配置中的 `device` 字段，支持 `"cuda:1"` 格式
- 检查配置中的 `gpu_id` 字段作为备用
- **关键修复**：调用 `torch.cuda.set_device(gpu_id)` 设置当前CUDA设备
- 添加设备验证，确保模型参数在正确的GPU上

### 3. 添加调试信息

添加了详细的调试输出：
- 显示设置的CUDA设备
- 验证模型参数所在设备
- 显示GPU内存使用情况

## 修复后的验证

### GPU使用情况
```
GPU 0: 26992 MiB, 28% 利用率（RNN训练）
GPU 1: 5537 MiB, 41% 利用率（Transformer训练）✅
```

### 日志输出
```
[DEBUG] 设置CUDA当前设备为: 1
[DEBUG] 验证设备: 实际使用GPU 1
[DEBUG] 单GPU模式: cuda:1
使用设备: cuda:1
[DEBUG] 模型构建前确认: 使用GPU 1
[DEBUG] 模型参数实际所在设备: cuda:1
[DEBUG] ✅ 设备匹配正确，模型在 cuda:1 上
[DEBUG] GPU1内存使用: 366.62 MB
```

## 使用方法

### 方法1：通过配置文件指定（推荐）

在配置文件中设置：
```yaml
device: "cuda:1"  # 直接指定GPU 1
```

或：
```yaml
device: "cuda"
gpu_id: 1  # 通过gpu_id指定
```

### 方法2：通过环境变量指定

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --config config/config_transformer.yaml
```

注意：使用 `CUDA_VISIBLE_DEVICES=1` 时，程序内部应使用 `cuda:0`（因为环境变量会重新映射GPU索引）。

## 关键代码修改

### train.py 中的关键修复

```python
# 确保设置当前CUDA设备（重要！）
if device.type == 'cuda':
    gpu_id = device.index if device.index is not None else 0
    torch.cuda.set_device(gpu_id)  # 这行很关键！
    print(f"[DEBUG] 设置CUDA当前设备为: {torch.cuda.current_device()}")
```

### 设备选择逻辑

```python
device_str = config.get('device', 'cuda')
if isinstance(device_str, str) and ':' in device_str:
    # 格式如 "cuda:1"
    device = torch.device(device_str)
elif config.get('gpu_id') is not None:
    # 配置中指定了gpu_id
    gpu_id = config.get('gpu_id')
    device = torch.device(f'cuda:{gpu_id}')
else:
    # 默认使用cuda:0
    device = torch.device('cuda:0')
```

## 验证方法

运行训练后，可以通过以下方式验证：

1. **nvidia-smi**：查看GPU使用情况
   ```bash
   nvidia-smi
   ```

2. **训练日志**：查看调试信息
   ```
   [DEBUG] ✅ 设备匹配正确，模型在 cuda:1 上
   [DEBUG] GPU1内存使用: XXX MB
   ```

3. **程序验证**：检查模型参数设备
   ```python
   first_param = next(model.parameters())
   print(first_param.device)  # 应该显示 cuda:1
   ```

## 注意事项

1. **torch.cuda.set_device() 的重要性**：
   - 仅仅创建 `torch.device('cuda:1')` 不够
   - 必须调用 `torch.cuda.set_device(1)` 来设置当前CUDA设备
   - 否则PyTorch可能仍然使用默认的GPU0

2. **数据加载器**：
   - DataLoader的 `pin_memory=True` 会将数据固定到CPU内存
   - 数据需要显式调用 `.to(device)` 移动到GPU
   - 代码中已经正确处理了这一点

3. **混合精度训练**：
   - `autocast('cuda')` 会使用当前设置的CUDA设备
   - 确保在设置设备后再创建autocast上下文

## 测试结果

✅ **问题已解决**：
- GPU1现在正常使用（5537 MiB内存，41%利用率）
- 模型参数在GPU1上
- 训练数据正确移动到GPU1
- 与GPU0上的RNN训练不冲突

