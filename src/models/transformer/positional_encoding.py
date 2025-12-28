"""
位置编码模块
实现绝对位置编码和相对位置编码
"""
import torch
import torch.nn as nn
import math


class AbsolutePositionalEncoding(nn.Module):
    """绝对位置编码（Absolute Positional Encoding）
    
    使用sin/cos函数生成位置编码，添加到词嵌入中
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout比例
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为buffer（不参与梯度更新）
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        Returns:
            x + positional_encoding: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """相对位置编码（Relative Positional Encoding）
    
    使用可学习的相对位置嵌入，在注意力计算中使用
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model: 模型维度
            max_len: 最大相对距离
        """
        super().__init__()
        self.max_len = max_len
        # 相对位置嵌入：[-max_len, max_len] -> [2*max_len+1, d_model]
        self.embeddings = nn.Embedding(2 * max_len + 1, d_model)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        生成相对位置编码
        
        Args:
            seq_len: 序列长度
        Returns:
            relative_pos: [seq_len, seq_len, d_model]
        """
        device = self.embeddings.weight.device
        # 创建相对位置矩阵
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        # 限制在[-max_len, max_len]范围内
        relative_positions = torch.clamp(relative_positions, -self.max_len, self.max_len)
        # 转换为索引 [0, 2*max_len]
        relative_positions = relative_positions + self.max_len
        # 获取嵌入
        relative_pos_emb = self.embeddings(relative_positions)
        return relative_pos_emb


class PositionalEncoding(nn.Module):
    """位置编码基类（默认使用绝对位置编码）"""
    
    def __init__(self, d_model: int, max_len: int = 5000, 
                 pos_encoding_type: str = 'absolute', dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            pos_encoding_type: 位置编码类型 ('absolute' 或 'relative')
            dropout: Dropout比例
        """
        super().__init__()
        self.pos_encoding_type = pos_encoding_type.lower()
        
        if self.pos_encoding_type == 'absolute':
            self.pos_encoding = AbsolutePositionalEncoding(d_model, max_len, dropout)
        elif self.pos_encoding_type == 'relative':
            self.pos_encoding = RelativePositionalEncoding(d_model, max_len)
        else:
            raise ValueError(f"未知的位置编码类型: {pos_encoding_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        Returns:
            添加位置编码后的张量
        """
        if self.pos_encoding_type == 'absolute':
            return self.pos_encoding(x)
        else:
            # 相对位置编码在注意力计算中使用，这里只返回原始输入
            return x
    
    def get_relative_pos_emb(self, seq_len: int) -> torch.Tensor:
        """
        获取相对位置编码（仅用于相对位置编码）
        
        Args:
            seq_len: 序列长度
        Returns:
            relative_pos_emb: [seq_len, seq_len, d_model]
        """
        if self.pos_encoding_type == 'relative':
            return self.pos_encoding(seq_len)
        else:
            raise ValueError("只有相对位置编码支持get_relative_pos_emb方法")


def create_positional_encoding(pos_encoding_type: str, d_model: int, 
                              max_len: int = 5000, dropout: float = 0.1) -> nn.Module:
    """
    创建位置编码
    
    Args:
        pos_encoding_type: 位置编码类型 ('absolute' 或 'relative')
        d_model: 模型维度
        max_len: 最大序列长度
        dropout: Dropout比例
        
    Returns:
        位置编码层
    """
    return PositionalEncoding(d_model, max_len, pos_encoding_type, dropout)


if __name__ == "__main__":
    # 测试位置编码
    batch_size, seq_len, d_model = 4, 20, 512
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 测试绝对位置编码
    abs_pos_enc = AbsolutePositionalEncoding(d_model)
    out = abs_pos_enc(x)
    print(f"绝对位置编码输出形状: {out.shape}")
    
    # 测试相对位置编码
    rel_pos_enc = RelativePositionalEncoding(d_model)
    rel_pos_emb = rel_pos_enc(seq_len)
    print(f"相对位置编码形状: {rel_pos_emb.shape}")

