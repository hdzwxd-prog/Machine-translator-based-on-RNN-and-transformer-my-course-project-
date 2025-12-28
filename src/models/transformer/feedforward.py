"""
前馈神经网络模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """前馈神经网络（Feed-Forward Network）
    
    两层线性变换，中间使用ReLU激活
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, 
                 activation: str = 'relu'):
        """
        Args:
            d_model: 模型维度
            d_ff: 前馈网络隐藏层维度（通常是d_model的4倍）
            dropout: Dropout比例
            activation: 激活函数 ('relu', 'gelu')
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation_type = activation.lower()
        
        if self.activation_type == 'relu':
            self.activation = F.relu
        elif self.activation_type == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"未知的激活函数: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    # 测试前馈网络
    batch_size, seq_len, d_model = 4, 10, 512
    d_ff = 2048
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    ff = FeedForward(d_model, d_ff)
    out = ff(x)
    print(f"前馈网络输出形状: {out.shape}")

