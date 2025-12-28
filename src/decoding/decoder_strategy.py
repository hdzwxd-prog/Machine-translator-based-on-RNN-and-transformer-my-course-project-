"""
解码策略模块
实现贪心解码和束搜索解码
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import heapq


class GreedyDecoder:
    """贪心解码器"""
    
    def __init__(self, 
                 model: nn.Module,
                 sos_idx: int,
                 eos_idx: int,
                 max_length: int = 100):
        """
        初始化贪心解码器
        
        Args:
            model: Seq2Seq模型
            sos_idx: 起始标记索引
            eos_idx: 结束标记索引
            max_length: 最大解码长度
        """
        self.model = model
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.max_length = max_length
    
    def decode(self, src: torch.Tensor, src_lengths: torch.Tensor) -> Tuple[List[int], List[torch.Tensor]]:
        """
        贪心解码
        
        Args:
            src: 源序列 [batch_size, src_len]
            src_lengths: 源序列长度 [batch_size]
            
        Returns:
            decoded_indices: 解码的词索引列表
            attention_weights: 注意力权重列表（Transformer模型可能为空）
        """
        self.model.eval()
        
        # 检测模型类型
        is_transformer = hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'pos_encoding')
        
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device
            
            if is_transformer:
                # Transformer模型解码
                src_mask = (src != 0).float()  # [batch_size, src_len]
                
                # 使用模型的generate方法（如果可用）或手动实现贪心解码
                if hasattr(self.model, 'generate'):
                    # 使用模型的generate方法
                    decoded_ids = self.model.generate(
                        src, src_mask, 
                        max_len=self.max_length,
                        sos_idx=self.sos_idx,
                        eos_idx=self.eos_idx,
                        strategy='greedy'
                    )
                    # decoded_ids: [batch_size, tgt_len]
                    decoded_indices = decoded_ids[0].tolist()
                    # 去掉padding和特殊token
                    decoded_indices = [idx for idx in decoded_indices if idx != 0 and idx != self.sos_idx]
                    if decoded_indices and decoded_indices[-1] == self.eos_idx:
                        decoded_indices = decoded_indices[:-1]
                    return decoded_indices, []
                else:
                    # 手动实现贪心解码
                    # 编码
                    encoder_output = self.model.encoder(src, src_mask)
                    
                    # 初始化解码器输入
                    decoder_input = torch.tensor([[self.sos_idx]] * batch_size).to(device)
                    decoded_indices = []
                    
                    for _ in range(self.max_length):
                        # 创建因果掩码
                        tgt_len = decoder_input.size(1)
                        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1)
                        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))
                        tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, 0.0)
                        tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)
                        
                        # 解码
                        decoder_output = self.model.decoder(decoder_input, encoder_output, tgt_mask, src_mask)
                        
                        # 输出投影
                        output = self.model.output_projection(decoder_output[:, -1, :])  # [batch_size, vocab_size]
                        
                        # 贪心选择
                        predicted = output.argmax(1)  # [batch_size]
                        
                        decoded_indices.append(predicted[0].item())
                        
                        # 如果遇到<eos>，停止解码
                        if predicted[0].item() == self.eos_idx:
                            break
                        
                        # 下一个输入
                        decoder_input = torch.cat([decoder_input, predicted.unsqueeze(1)], dim=1)
                    
                    return decoded_indices, []
            else:
                # RNN模型解码（原有逻辑）
                # 编码
                encoder_outputs, encoder_hidden = self.model.encoder(src, src_lengths)
                
                # 初始化解码器隐藏状态
                decoder_hidden = self.model._init_decoder_hidden(encoder_hidden)
                
                # 创建源序列掩码
                src_mask = self.model._create_mask(src)
                
                # 第一个输入是<sos>
                decoder_input = torch.tensor([[self.sos_idx]] * batch_size).to(device)
                
                decoded_indices = []
                attention_weights_list = []
                
                for _ in range(self.max_length):
                    # 解码一步
                    output, decoder_hidden, attn_weights = self.model.decoder(
                        decoder_input,
                        decoder_hidden,
                        encoder_outputs,
                        src_mask
                    )
                    
                    # 贪心选择：选择概率最大的词
                    predicted = output.argmax(1)
                    
                    decoded_indices.append(predicted.item())
                    attention_weights_list.append(attn_weights)
                    
                    # 如果遇到<eos>，停止解码
                    if predicted.item() == self.eos_idx:
                        break
                    
                    # 下一个输入
                    decoder_input = predicted.unsqueeze(1)
                
                return decoded_indices, attention_weights_list


class BeamSearchDecoder:
    """束搜索解码器"""
    
    def __init__(self,
                 model: nn.Module,
                 sos_idx: int,
                 eos_idx: int,
                 beam_size: int = 5,
                 max_length: int = 100,
                 length_penalty: float = 0.6):
        """
        初始化束搜索解码器
        
        Args:
            model: Seq2Seq模型
            sos_idx: 起始标记索引
            eos_idx: 结束标记索引
            beam_size: 束大小
            max_length: 最大解码长度
            length_penalty: 长度惩罚系数
        """
        self.model = model
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
    
    def decode(self, src: torch.Tensor, src_lengths: torch.Tensor) -> Tuple[List[int], List[torch.Tensor]]:
        """
        束搜索解码
        
        Args:
            src: 源序列 [1, src_len]（单个样本）
            src_lengths: 源序列长度 [1]
            
        Returns:
            decoded_indices: 解码的词索引列表（最佳路径）
            attention_weights: 注意力权重列表
        """
        self.model.eval()
        
        # 检测模型类型
        is_transformer = hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'pos_encoding')
        
        with torch.no_grad():
            device = src.device
            
            # 确保输入在正确的设备上
            if not src.is_cuda and next(self.model.parameters()).is_cuda:
                device = next(self.model.parameters()).device
                src = src.to(device)
                if src_lengths is not None:
                    src_lengths = src_lengths.to(device)
            
            if is_transformer:
                # Transformer模型：使用模型的generate方法
                src_mask = (src != 0).float()  # [batch_size, src_len]
                
                if hasattr(self.model, 'generate'):
                    # 使用模型的generate方法进行beam search
                    decoded_ids = self.model.generate(
                        src, src_mask,
                        max_len=self.max_length,
                        sos_idx=self.sos_idx,
                        eos_idx=self.eos_idx,
                        strategy='beam_search',
                        beam_size=self.beam_size
                    )
                    # decoded_ids: [batch_size, tgt_len]
                    decoded_indices = decoded_ids[0].tolist()
                    # 去掉padding和特殊token
                    decoded_indices = [idx for idx in decoded_indices if idx != 0 and idx != self.sos_idx]
                    if decoded_indices and decoded_indices[-1] == self.eos_idx:
                        decoded_indices = decoded_indices[:-1]
                    return decoded_indices, []
                else:
                    # 如果模型没有generate方法，手动实现beam search
                    # 编码
                    encoder_output = self.model.encoder(src, src_mask)
                    
                    # 初始化beam: (序列tensor, 分数)
                    beam = [(torch.tensor([[self.sos_idx]], device=device), 0.0)]
                    
                    for step in range(self.max_length):
                        candidates = []
                        
                        for seq, score in beam:
                            # 如果已经生成<eos>，加入完成列表
                            if seq[0, -1].item() == self.eos_idx:
                                candidates.append((seq, score))
                                continue
                            
                            # 创建因果掩码
                            tgt_len = seq.size(1)
                            tgt_mask = self.model._generate_square_subsequent_mask(tgt_len).to(device)
                            tgt_mask = tgt_mask.unsqueeze(0)
                            
                            # 解码
                            decoder_output = self.model.decoder(seq, encoder_output, tgt_mask, src_mask)
                            next_token_logits = self.model.output_projection(decoder_output[:, -1, :])
                            next_token_probs = torch.softmax(next_token_logits, dim=-1)
                            
                            # 获取top-k
                            top_probs, top_indices = next_token_probs.topk(self.beam_size, dim=-1)
                            
                            for i in range(self.beam_size):
                                next_token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                                new_seq = torch.cat([seq, next_token], dim=1)
                                new_score = score + torch.log(top_probs[0, i] + 1e-10).item()
                                candidates.append((new_seq, new_score))
                        
                        # 选择top-k候选
                        candidates.sort(key=lambda x: x[1] / (len(x[0][0]) ** self.length_penalty), reverse=True)
                        beam = candidates[:self.beam_size]
                        
                        # 检查是否都结束
                        if all(seq[0, -1].item() == self.eos_idx for seq, _ in beam):
                            break
                    
                    # 选择得分最高的序列
                    best_seq = beam[0][0]
                    decoded_indices = best_seq[0, 1:].tolist()  # 去掉<sos>
                    # 去掉<eos>
                    if decoded_indices and decoded_indices[-1] == self.eos_idx:
                        decoded_indices = decoded_indices[:-1]
                    return decoded_indices, []
            
            else:
                # RNN模型解码（原有逻辑）
                # 编码
                encoder_outputs, encoder_hidden = self.model.encoder(src, src_lengths)
                # encoder_outputs: [1, src_len, hidden_dim]
                
                # 初始化解码器隐藏状态
                decoder_hidden = self.model._init_decoder_hidden(encoder_hidden)
                
                # 创建源序列掩码
                src_mask = self.model._create_mask(src)
            
            # 初始化束
            # 每个beam: (score, tokens, hidden, attention_weights)
            beams = [(0.0, [self.sos_idx], decoder_hidden, [])]
            completed_beams = []
            
            for step in range(self.max_length):
                candidates = []
                
                for score, tokens, hidden, attn_list in beams:
                    # 如果已经生成<eos>，加入完成列表
                    if tokens[-1] == self.eos_idx:
                        completed_beams.append((score, tokens, attn_list))
                        continue
                    
                    # 解码一步
                    decoder_input = torch.tensor([[tokens[-1]]]).to(device)
                    
                    output, new_hidden, attn_weights = self.model.decoder(
                        decoder_input,
                        hidden,
                        encoder_outputs,
                        src_mask
                    )
                    
                    # 计算概率
                    log_probs = F.log_softmax(output, dim=-1)  # [1, vocab_size]
                    
                    # 获取top-k候选
                    topk_log_probs, topk_indices = log_probs.topk(self.beam_size)
                    
                    # 扩展beam
                    for k in range(self.beam_size):
                        token = topk_indices[0, k].item()
                        token_log_prob = topk_log_probs[0, k].item()
                        
                        new_score = score + token_log_prob
                        new_tokens = tokens + [token]
                        new_attn_list = attn_list + [attn_weights]
                        
                        candidates.append((new_score, new_tokens, new_hidden, new_attn_list))
                
                # 如果没有候选，停止
                if not candidates:
                    break
                
                # 选择top-k候选作为新的beams
                # 使用长度归一化
                beams = sorted(candidates, key=lambda x: x[0] / (len(x[1]) ** self.length_penalty), reverse=True)[:self.beam_size]
                
                # 如果所有beam都完成了，停止
                if all(tokens[-1] == self.eos_idx for _, tokens, _, _ in beams):
                    for beam in beams:
                        if beam not in completed_beams:
                            completed_beams.append((beam[0], beam[1], beam[3]))
                    break
            
            # 选择最佳完成序列
            if not completed_beams:
                # 如果没有完成的序列，选择当前最佳beam
                best_beam = beams[0]
                best_tokens = best_beam[1][1:]  # 去掉<sos>
                best_attn = best_beam[3]
            else:
                # 选择得分最高的完成序列
                best_beam = max(completed_beams, key=lambda x: x[0] / (len(x[1]) ** self.length_penalty))
                best_tokens = best_beam[1][1:]  # 去掉<sos>
                best_attn = best_beam[2]
            
            # 去掉<eos>
            if best_tokens and best_tokens[-1] == self.eos_idx:
                best_tokens = best_tokens[:-1]
            
            return best_tokens, best_attn


if __name__ == "__main__":
    # 这里是测试代码，需要实际的模型才能运行
    print("解码策略模块")

