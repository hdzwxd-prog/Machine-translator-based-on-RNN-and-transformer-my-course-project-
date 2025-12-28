"""
诊断脚本：分析为什么loss很小但预测结果很差
"""
import os
import sys
import torch
import yaml
import numpy as np
from torch.nn import functional as F

sys.path.insert(0, '.')

from train import load_config, build_model, prepare_data
from src.data.vocab import Vocabulary
from src.training.trainer import Trainer


def diagnose_loss():
    """诊断loss计算问题"""
    print("=" * 80)
    print("Loss诊断分析")
    print("=" * 80)
    
    # 加载配置
    config = load_config('config/config_transformer.yaml')
    
    # 设置设备
    device_str = config.get('device', 'cuda')
    if isinstance(device_str, str) and ':' in device_str:
        device = torch.device(device_str)
    elif config.get('gpu_id') is not None:
        device = torch.device(f'cuda:{config.get("gpu_id")}')
    else:
        device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        gpu_id = device.index if device.index is not None else 0
        torch.cuda.set_device(gpu_id)
    
    print(f"\n使用设备: {device}")
    
    # 准备数据
    print("\n准备数据...")
    use_cache = config['training'].get('use_data_cache', True)
    train_loader, valid_loader, src_vocab, tgt_vocab = prepare_data(config, use_cache=use_cache)
    
    # 构建模型
    print("\n构建模型...")
    model = build_model(config, src_vocab, tgt_vocab, device)
    
    # 加载checkpoint
    checkpoint_path = 'checkpoints_transformer/best_model.pt'
    print(f"\n加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint valid loss: {checkpoint['best_valid_loss']:.6f}")
    
    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt_vocab.PAD_IDX, reduction='mean')
    
    # 获取一个验证batch
    print("\n" + "=" * 80)
    print("获取验证batch并计算loss")
    print("=" * 80)
    
    valid_iter = iter(valid_loader)
    src, src_lengths, tgt, tgt_lengths = next(valid_iter)
    
    print(f"\nBatch信息:")
    print(f"  src形状: {src.shape}")
    print(f"  tgt形状: {tgt.shape}")
    print(f"  src_lengths: {src_lengths.tolist()}")
    print(f"  tgt_lengths: {tgt_lengths.tolist()}")
    
    # 移动到设备
    src = src.to(device)
    src_lengths = src_lengths.to(device)
    tgt = tgt.to(device)
    
    # 创建掩码
    src_mask = (src != 0).float()
    tgt_mask = (tgt != 0).float()
    
    print(f"\n掩码信息:")
    print(f"  src_mask形状: {src_mask.shape}")
    print(f"  tgt_mask形状: {tgt_mask.shape}")
    print(f"  src_mask非零元素数: {(src_mask != 0).sum().item()}")
    print(f"  tgt_mask非零元素数: {(tgt_mask != 0).sum().item()}")
    
    # 前向传播 - 使用teacher forcing（训练时的模式）
    print("\n" + "=" * 80)
    print("前向传播（使用Teacher Forcing，teacher_forcing_ratio=1.0）")
    print("=" * 80)
    
    with torch.no_grad():
        outputs_teacher = model(src, tgt, src_mask, tgt_mask, teacher_forcing_ratio=1.0)
        
    print("\n" + "=" * 80)
    print("前向传播（不使用Teacher Forcing，teacher_forcing_ratio=0.0）")
    print("=" * 80)
    
    with torch.no_grad():
        outputs_no_teacher = model(src, tgt, src_mask, tgt_mask, teacher_forcing_ratio=0.0)
    
    # 使用teacher forcing的结果进行分析
    outputs = outputs_teacher
    
    print(f"\n模型输出（Teacher Forcing）:")
    print(f"  outputs形状: {outputs.shape}")
    print(f"  outputs范围: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}")
    
    # 准备target（去掉<sos>）
    tgt_flat = tgt[:, 1:].contiguous().view(-1)  # [batch_size * (tgt_len-1)]
    outputs_flat = outputs.contiguous().view(-1, outputs.size(-1))  # [batch_size * (tgt_len-1), vocab_size]
    
    print(f"\n展平后:")
    print(f"  outputs_flat形状: {outputs_flat.shape}")
    print(f"  tgt_flat形状: {tgt_flat.shape}")
    print(f"  tgt_flat中PAD的数量: {(tgt_flat == tgt_vocab.PAD_IDX).sum().item()}")
    print(f"  tgt_flat中非PAD的数量: {(tgt_flat != tgt_vocab.PAD_IDX).sum().item()}")
    
    # 计算loss
    loss = criterion(outputs_flat, tgt_flat)
    print(f"\n总体Loss: {loss.item():.6f}")
    
    # 详细分析每个样本
    print("\n" + "=" * 80)
    print("详细分析每个样本")
    print("=" * 80)
    
    batch_size = src.size(0)
    for i in range(min(3, batch_size)):  # 只分析前3个样本
        print(f"\n样本 {i+1}:")
    print("-" * 80)
    
    # 源句子
    src_tokens = src_vocab.decode(src[i].cpu().tolist(), skip_special=True)
    print(f"源句子: {' '.join(src_tokens)}")
    
    # 目标句子（真值）
    tgt_tokens = tgt_vocab.decode(tgt[i].cpu().tolist(), skip_special=True)
    print(f"真值: {' '.join(tgt_tokens)}")
    
    # 模型预测（使用argmax）
    pred_logits = outputs[i]  # [tgt_len-1, vocab_size]
    pred_indices = pred_logits.argmax(dim=-1).cpu().tolist()  # [tgt_len-1]
    pred_tokens = tgt_vocab.decode(pred_indices, skip_special=True)
    print(f"预测: {' '.join(pred_tokens)}")
    
    # 计算这个样本的loss
    sample_tgt = tgt[i, 1:]  # [tgt_len-1]
    sample_outputs = outputs[i]  # [tgt_len-1, vocab_size]
    sample_loss = criterion(sample_outputs, sample_tgt)
    print(f"样本Loss: {sample_loss.item():.6f}")
    
    # 分析每个时间步
    print(f"\n  时间步分析:")
    actual_tgt_len = tgt_lengths[i].item()
    print(f"  实际目标长度: {actual_tgt_len}")
    
    for t in range(min(actual_tgt_len - 1, 10)):  # 只显示前10个时间步
        tgt_token_idx = sample_tgt[t].item()
        pred_token_idx = pred_indices[t]
        
        # 获取logits
        logits = sample_outputs[t]  # [vocab_size]
        probs = F.softmax(logits, dim=-1)
        
        # 真值token
        tgt_token = tgt_vocab.idx2word.get(tgt_token_idx, '<UNK>')
        tgt_prob = probs[tgt_token_idx].item()
        
        # 预测token
        pred_token = tgt_vocab.idx2word.get(pred_token_idx, '<UNK>')
        pred_prob = probs[pred_token_idx].item()
        
        # 计算这个时间步的loss
        step_loss = F.cross_entropy(
            logits.unsqueeze(0),
            torch.tensor([tgt_token_idx], device=device),
            ignore_index=tgt_vocab.PAD_IDX,
            reduction='none'
        ).item()
        
        # 检查是否匹配
        match = "✓" if tgt_token_idx == pred_token_idx else "✗"
        
        print(f"    t={t}: 真值={tgt_token}({tgt_token_idx}, prob={tgt_prob:.4f}) | "
              f"预测={pred_token}({pred_token_idx}, prob={pred_prob:.4f}) | "
              f"loss={step_loss:.6f} {match}")
        
        # 如果是PAD，跳过后续
        if tgt_token_idx == tgt_vocab.PAD_IDX:
            break

# 分析loss分布
print("\n" + "=" * 80)
print("Loss分布分析")
print("=" * 80)

# 计算每个时间步的loss（不考虑PAD）
step_losses = []
for i in range(batch_size):
    sample_tgt = tgt[i, 1:]
    sample_outputs = outputs[i]
    actual_len = tgt_lengths[i].item() - 1  # 减去1因为去掉了<sos>
    
    for t in range(actual_len):
        if sample_tgt[t].item() != tgt_vocab.PAD_IDX:
            step_loss = F.cross_entropy(
                sample_outputs[t:t+1],
                sample_tgt[t:t+1],
                ignore_index=tgt_vocab.PAD_IDX,
                reduction='none'
            ).item()
            step_losses.append(step_loss)

if step_losses:
    step_losses = np.array(step_losses)
    print(f"\n时间步loss统计（共{len(step_losses)}个非PAD时间步）:")
    print(f"  平均loss: {step_losses.mean():.6f}")
    print(f"  中位数loss: {np.median(step_losses):.6f}")
    print(f"  最小loss: {step_losses.min():.6f}")
    print(f"  最大loss: {step_losses.max():.6f}")
    print(f"  标准差: {step_losses.std():.6f}")
    
    # 分析loss的分布
    print(f"\nLoss分布:")
    print(f"  loss < 0.001: {(step_losses < 0.001).sum()} ({100*(step_losses < 0.001).sum()/len(step_losses):.1f}%)")
    print(f"  loss < 0.01: {(step_losses < 0.01).sum()} ({100*(step_losses < 0.01).sum()/len(step_losses):.1f}%)")
    print(f"  loss < 0.1: {(step_losses < 0.1).sum()} ({100*(step_losses < 0.1).sum()/len(step_losses):.1f}%)")
    print(f"  loss < 1.0: {(step_losses < 1.0).sum()} ({100*(step_losses < 1.0).sum()/len(step_losses):.1f}%)")
    print(f"  loss >= 1.0: {(step_losses >= 1.0).sum()} ({100*(step_losses >= 1.0).sum()/len(step_losses):.1f}%)")

# 分析预测准确率（Teacher Forcing）
print("\n" + "=" * 80)
print("预测准确率分析（Teacher Forcing模式）")
print("=" * 80)

correct = 0
total = 0
for i in range(batch_size):
    sample_tgt = tgt[i, 1:]
    sample_outputs = outputs[i]
    pred_indices = sample_outputs.argmax(dim=-1)
    actual_len = tgt_lengths[i].item() - 1
    
            for t in range(actual_len):
                if sample_tgt[t].item() != tgt_vocab.PAD_IDX:
                    total += 1
                    if sample_tgt[t].item() == pred_indices[t].item():
                        correct += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"\nToken级别准确率（Teacher Forcing）: {correct}/{total} = {accuracy*100:.2f}%")
        
        # 分析预测准确率（不使用Teacher Forcing）
        print("\n" + "=" * 80)
        print("预测准确率分析（不使用Teacher Forcing模式）")
        print("=" * 80)
        
        correct_no_teacher = 0
        total_no_teacher = 0
        for i in range(batch_size):
            sample_tgt = tgt[i, 1:]
            sample_outputs = outputs_no_teacher[i]
            pred_indices = sample_outputs.argmax(dim=-1)
            actual_len = tgt_lengths[i].item() - 1
            
            for t in range(actual_len):
                if sample_tgt[t].item() != tgt_vocab.PAD_IDX:
                    total_no_teacher += 1
                    if sample_tgt[t].item() == pred_indices[t].item():
                        correct_no_teacher += 1
        
        accuracy_no_teacher = correct_no_teacher / total_no_teacher if total_no_teacher > 0 else 0
        print(f"\nToken级别准确率（不使用Teacher Forcing）: {correct_no_teacher}/{total_no_teacher} = {accuracy_no_teacher*100:.2f}%")
        
        # 比较两个模式的差异
        print("\n" + "=" * 80)
        print("Teacher Forcing vs 非Teacher Forcing 对比")
        print("=" * 80)
        
        # 使用generate方法进行自回归生成（这是实际推理时使用的方法）
        print("\n使用generate方法进行自回归生成（实际推理模式）:")
        from src.decoding import GreedyDecoder
        decoder = GreedyDecoder(
            model=model,
            sos_idx=tgt_vocab.SOS_IDX,
            eos_idx=tgt_vocab.EOS_IDX,
            max_length=100
        )
        
        print("\n样本1的自回归生成结果:")
        src_single = src[0:1]
        src_lengths_single = src_lengths[0:1]
        decoded_indices, _ = decoder.decode(src_single, src_lengths_single)
        decoded_tokens = tgt_vocab.decode(decoded_indices, skip_special=True)
        print(f"  源句子: {' '.join(src_vocab.decode(src[0].cpu().tolist(), skip_special=True))}")
        print(f"  真值: {' '.join(tgt_vocab.decode(tgt[0].cpu().tolist(), skip_special=True))}")
        print(f"  自回归生成: {' '.join(decoded_tokens)}")
        
        # 分析最常预测的词
        print("\n" + "=" * 80)
        print("预测词频分析")
        print("=" * 80)
        
        pred_word_counts = {}
        for i in range(batch_size):
            pred_indices = outputs[i].argmax(dim=-1).cpu().tolist()
            for idx in pred_indices:
                word = tgt_vocab.idx2word.get(idx, '<UNK>')
                pred_word_counts[word] = pred_word_counts.get(word, 0) + 1
        
        sorted_words = sorted(pred_word_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop-10预测词:")
        for word, count in sorted_words[:10]:
            print(f"  {word}: {count}次")


if __name__ == "__main__":
    diagnose_loss()

