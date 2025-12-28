"""
Transformer编码器模块
"""
import math
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import FeedForward
from .normalization import create_normalization


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, norm_type: str = 'layernorm',
                 use_relative_pos: bool = False, max_len: int = 5000):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: Dropout比例
            norm_type: 归一化类型 ('layernorm' 或 'rmsnorm')
            use_relative_pos: 是否使用相对位置编码
            max_len: 最大序列长度
        """
        super().__init__()
        
        # 自注意力
        self.self_attn = MultiHeadAttention(
            d_model, num_heads, dropout, use_relative_pos, max_len
        )
        
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # 归一化层
        self.norm1 = create_normalization(norm_type, d_model)
        self.norm2 = create_normalization(norm_type, d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                relative_pos_emb: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] 或 [batch_size, 1, seq_len]
            relative_pos_emb: [seq_len, seq_len, d_k] 相对位置编码（可选）
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # 自注意力 + 残差连接 + 归一化
        attn_output, _ = self.self_attn(x, x, x, mask, relative_pos_emb)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络 + 残差连接 + 归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_layers: int, d_ff: int, max_len: int = 5000,
                 dropout: float = 0.1, padding_idx: int = 0,
                 pos_encoding_type: str = 'absolute',
                 norm_type: str = 'layernorm',
                 use_relative_pos: bool = False):
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: 编码器层数
            d_ff: 前馈网络隐藏层维度
            max_len: 最大序列长度
            dropout: Dropout比例
            padding_idx: Padding索引
            pos_encoding_type: 位置编码类型 ('absolute' 或 'relative')
            norm_type: 归一化类型 ('layernorm' 或 'rmsnorm')
            use_relative_pos: 是否使用相对位置编码
        """
        super().__init__()
        
        self.d_model = d_model
        self.use_relative_pos = use_relative_pos
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        
        # 位置编码
        from .positional_encoding import create_positional_encoding
        self.pos_encoding = create_positional_encoding(
            pos_encoding_type, d_model, max_len, dropout
        )
        
        # 编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, num_heads, d_ff, dropout, norm_type,
                use_relative_pos, max_len
            )
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            src: 源序列 [batch_size, src_len]
            src_mask: 源序列掩码 [batch_size, src_len] 或 [batch_size, 1, src_len]
            
        Returns:
            output: [batch_size, src_len, d_model]
        """
        # 词嵌入
        x = self.embedding(src) * math.sqrt(self.d_model)
        
        # 位置编码
        if self.use_relative_pos:
            x = self.pos_encoding(x)
            # 获取相对位置编码
            seq_len = src.size(1)
            relative_pos_emb = self.pos_encoding.get_relative_pos_emb(seq_len)
        else:
            x = self.pos_encoding(x)
            relative_pos_emb = None
        
        x = self.dropout(x)
        
        # 创建注意力掩码
        if src_mask is not None:
            if src_mask.dim() == 2:
                # [batch_size, src_len] -> [batch_size, 1, 1, src_len]
                src_mask = src_mask.unsqueeze(1).unsqueeze(2)
            elif src_mask.dim() == 3:
                # [batch_size, 1, src_len] -> [batch_size, 1, 1, src_len]
                src_mask = src_mask.unsqueeze(1)
        
        # 通过编码器层
        for layer in self.layers:
            x = layer(x, src_mask, relative_pos_emb)
        
        return x

