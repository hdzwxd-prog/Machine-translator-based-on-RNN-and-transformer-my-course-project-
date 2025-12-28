"""
Seq2Seq模型
整合编码器和解码器
"""
import torch
import torch.nn as nn
from .encoder import RNNEncoder
from .decoder import RNNDecoder


class Seq2Seq(nn.Module):
    """Seq2Seq模型"""
    
    def __init__(self, encoder: RNNEncoder, decoder: RNNDecoder):
        """
        初始化Seq2Seq模型
        
        Args:
            encoder: 编码器
            decoder: 解码器
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        # 如果编码器和解码器维度不同，需要转换层
        if encoder.hidden_dim * encoder.num_directions != decoder.hidden_dim:
            self.bridge = nn.Linear(
                encoder.hidden_dim * encoder.num_directions,
                decoder.hidden_dim
            )
        else:
            self.bridge = None
    
    def _init_decoder_hidden(self, encoder_hidden):
        """
        将编码器的隐藏状态转换为解码器的初始隐藏状态
        
        Args:
            encoder_hidden: 编码器隐藏状态
            
        Returns:
            解码器初始隐藏状态
        """
        if self.encoder.cell_type == 'lstm':
            h, c = encoder_hidden
        else:
            h = encoder_hidden
        
        # 如果是双向编码器，需要合并前向和后向
        if self.encoder.bidirectional:
            # h: [num_layers * 2, batch_size, hidden_dim]
            # 将前向和后向拼接
            def combine_directions(hidden):
                # hidden: [num_layers * 2, batch_size, hidden_dim]
                # -> [num_layers, batch_size, hidden_dim * 2]
                batch_size = hidden.size(1)
                hidden = hidden.view(
                    self.encoder.num_layers, 2, batch_size, self.encoder.hidden_dim
                )
                hidden = hidden.transpose(1, 2).contiguous()
                hidden = hidden.view(
                    self.encoder.num_layers, batch_size, self.encoder.hidden_dim * 2
                )
                return hidden
            
            h = combine_directions(h)
            if self.encoder.cell_type == 'lstm':
                c = combine_directions(c)
        
        # 如果需要维度转换
        if self.bridge is not None:
            h = self.bridge(h)
            if self.encoder.cell_type == 'lstm':
                c = self.bridge(c)
        
        # 返回解码器隐藏状态
        if self.decoder.cell_type == 'lstm':
            return (h, c)
        else:
            return h
    
    def forward(self, 
                src: torch.Tensor,
                src_lengths: torch.Tensor,
                tgt: torch.Tensor,
                teacher_forcing_ratio: float = 0.5):
        """
        前向传播
        
        Args:
            src: 源序列 [batch_size, src_len]
            src_lengths: 源序列长度 [batch_size]
            tgt: 目标序列 [batch_size, tgt_len]（包含<sos>）
            teacher_forcing_ratio: Teacher Forcing比例
            
        Returns:
            outputs: 输出 [batch_size, tgt_len-1, vocab_size]
            attention_weights: 注意力权重 [batch_size, tgt_len-1, src_len]
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.vocab_size
        
        # 编码
        encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)
        # encoder_outputs: [batch_size, src_len, encoder_hidden_dim]
        
        # 初始化解码器隐藏状态
        decoder_hidden = self._init_decoder_hidden(encoder_hidden)
        
        # 创建源序列掩码
        src_mask = self._create_mask(src)
        
        # 存储输出
        outputs = torch.zeros(batch_size, tgt_len - 1, vocab_size).to(src.device)
        attention_weights_list = []
        
        # 第一个输入是<sos>
        decoder_input = tgt[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        # 逐步解码
        for t in range(1, tgt_len):
            # 解码一步
            output, decoder_hidden, attn_weights = self.decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs,
                src_mask
            )
            
            # 存储输出
            outputs[:, t-1, :] = output
            attention_weights_list.append(attn_weights)
            
            # Teacher Forcing：决定下一个输入
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            if teacher_force:
                # 使用真实的下一个词
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                # 使用模型预测的词
                decoder_input = output.argmax(1).unsqueeze(1)
        
        # 堆叠注意力权重
        attention_weights = torch.stack(attention_weights_list, dim=1)
        # [batch_size, tgt_len-1, src_len]
        
        return outputs, attention_weights
    
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
    # 测试Seq2Seq模型
    src_vocab_size = 10000
    tgt_vocab_size = 8000
    embed_dim = 256
    hidden_dim = 512
    batch_size = 4
    src_len = 15
    tgt_len = 12
    
    encoder = RNNEncoder(
        vocab_size=src_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        cell_type='lstm'
    )
    
    decoder = RNNDecoder(
        vocab_size=tgt_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        cell_type='lstm',
        attention_type='dot',
        encoder_hidden_dim=hidden_dim
    )
    
    model = Seq2Seq(encoder, decoder)
    
    # 随机输入
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    src_lengths = torch.randint(5, src_len+1, (batch_size,))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))
    
    # 前向传播
    outputs, attention_weights = model(src, src_lengths, tgt, teacher_forcing_ratio=0.5)
    
    print(f"输出: {outputs.shape}")
    print(f"注意力权重: {attention_weights.shape}")

