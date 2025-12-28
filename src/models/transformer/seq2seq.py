"""
Transformer Seq2Seq模型
"""
import torch
import torch.nn as nn
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class TransformerSeq2Seq(nn.Module):
    """基于Transformer的Seq2Seq模型"""
    
    def __init__(self, encoder: TransformerEncoder, decoder: TransformerDecoder):
        """
        初始化Transformer Seq2Seq模型
        
        Args:
            encoder: Transformer编码器
            decoder: Transformer解码器
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        # 输出投影层
        self.output_projection = nn.Linear(
            decoder.d_model, 
            decoder.embedding.num_embeddings
        )
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None,
                teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        """
        前向传播（训练模式）
        
        Args:
            src: 源序列 [batch_size, src_len]
            tgt: 目标序列 [batch_size, tgt_len]（包含<sos>）
            src_mask: 源序列掩码 [batch_size, src_len]
            tgt_mask: 目标序列掩码 [batch_size, tgt_len]
            teacher_forcing_ratio: Teacher Forcing比例
            
        Returns:
            outputs: 输出 [batch_size, tgt_len-1, vocab_size]
        """
        # 编码
        encoder_output = self.encoder(src, src_mask)
        # encoder_output: [batch_size, src_len, d_model]
        
        # 解码
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        # decoder_output: [batch_size, tgt_len, d_model]
        
        # 输出投影（去掉最后一个时间步，因为不需要预测<sos>之后的内容）
        outputs = self.output_projection(decoder_output[:, :-1, :])
        # outputs: [batch_size, tgt_len-1, vocab_size]
        
        return outputs
    
    def generate(self, src: torch.Tensor, src_mask: torch.Tensor,
                 max_len: int = 100, sos_idx: int = 1, eos_idx: int = 2,
                 strategy: str = 'greedy', beam_size: int = 5) -> torch.Tensor:
        """
        生成翻译（推理模式）
        
        Args:
            src: 源序列 [batch_size, src_len]
            src_mask: 源序列掩码 [batch_size, src_len]
            max_len: 最大生成长度
            sos_idx: <sos>索引
            eos_idx: <eos>索引
            strategy: 解码策略 ('greedy' 或 'beam_search')
            beam_size: Beam search大小
            
        Returns:
            translations: [batch_size, tgt_len]
        """
        batch_size = src.size(0)
        device = src.device
        
        # 编码
        encoder_output = self.encoder(src, src_mask)
        
        if strategy == 'greedy':
            # 贪心解码
            translations = []
            decoder_input = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
            
            for _ in range(max_len):
                # 创建因果掩码
                tgt_len = decoder_input.size(1)
                tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(device)
                tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)
                
                # 解码
                decoder_output = self.decoder(decoder_input, encoder_output, tgt_mask, src_mask)
                # decoder_output: [batch_size, tgt_len, d_model]
                
                # 获取最后一个时间步的预测
                next_token_logits = self.output_projection(decoder_output[:, -1, :])
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                
                # 添加到序列
                decoder_input = torch.cat([decoder_input, next_token], dim=1)
                
                # 检查是否所有序列都结束
                if (next_token == eos_idx).all():
                    break
            
            return decoder_input[:, 1:]  # 去掉<sos>
        
        elif strategy == 'beam_search':
            # Beam search解码（简化版）
            # 这里实现一个简单的beam search
            translations = []
            
            for b in range(batch_size):
                # 对每个样本单独进行beam search
                beam = [(torch.tensor([[sos_idx]], device=device), 0.0)]
                encoder_output_b = encoder_output[b:b+1]
                src_mask_b = src_mask[b:b+1] if src_mask is not None else None
                
                for step in range(max_len):
                    candidates = []
                    
                    for seq, score in beam:
                        if seq[0, -1].item() == eos_idx:
                            candidates.append((seq, score))
                            continue
                        
                        # 创建掩码
                        tgt_len = seq.size(1)
                        tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(device)
                        tgt_mask = tgt_mask.unsqueeze(0)
                        
                        # 解码
                        decoder_output = self.decoder(seq, encoder_output_b, tgt_mask, src_mask_b)
                        next_token_logits = self.output_projection(decoder_output[:, -1, :])
                        next_token_probs = torch.softmax(next_token_logits, dim=-1)
                        
                        # 获取top-k
                        top_probs, top_indices = next_token_probs.topk(beam_size, dim=-1)
                        
                        for i in range(beam_size):
                            next_token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                            new_seq = torch.cat([seq, next_token], dim=1)
                            new_score = score + torch.log(top_probs[0, i] + 1e-10).item()
                            candidates.append((new_seq, new_score))
                    
                    # 选择top-k
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    beam = candidates[:beam_size]
                    
                    # 检查是否都结束
                    if all(seq[0, -1].item() == eos_idx for seq, _ in beam):
                        break
                
                # 选择得分最高的序列
                best_seq = beam[0][0]
                translations.append(best_seq[0, 1:])  # 去掉<sos>
            
            # 填充到相同长度
            max_tgt_len = max(len(t) for t in translations)
            padded_translations = []
            for t in translations:
                if len(t) < max_tgt_len:
                    padding = torch.zeros(max_tgt_len - len(t), dtype=torch.long, device=device)
                    t = torch.cat([t, padding])
                padded_translations.append(t)
            
            return torch.stack(padded_translations, dim=0)
        
        else:
            raise ValueError(f"未知的解码策略: {strategy}")
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """生成因果掩码（下三角矩阵）"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def _create_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        创建源序列掩码（mask掉padding）
        
        Args:
            src: 源序列 [batch_size, src_len]
            
        Returns:
            mask: [batch_size, src_len]
        """
        # 0是padding索引
        mask = (src != 0).float()
        return mask


if __name__ == "__main__":
    # 测试Transformer Seq2Seq模型
    src_vocab_size = 10000
    tgt_vocab_size = 8000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    batch_size = 4
    src_len = 15
    tgt_len = 12
    
    encoder = TransformerEncoder(
        vocab_size=src_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff
    )
    
    decoder = TransformerDecoder(
        vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff
    )
    
    model = TransformerSeq2Seq(encoder, decoder)
    
    # 随机输入
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))
    src_mask = (src != 0).float()
    tgt_mask = (tgt != 0).float()
    
    # 前向传播
    outputs = model(src, tgt, src_mask, tgt_mask, teacher_forcing_ratio=1.0)
    
    print(f"输出形状: {outputs.shape}")

