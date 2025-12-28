"""
归一化模块
实现LayerNorm和RMSNorm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """层归一化（Layer Normalization）"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model: 模型维度
            eps: 防止除零的小值
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            normalized: [batch_size, seq_len, d_model]
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class RMSNorm(nn.Module):
    """均方根归一化（Root Mean Square Normalization）
    
    RMSNorm是LayerNorm的简化版本，只使用RMS而不减去均值
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model: 模型维度
            eps: 防止除零的小值
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            normalized: [batch_size, seq_len, d_model]
        """
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * x / rms


def create_normalization(norm_type: str, d_model: int, eps: float = 1e-6) -> nn.Module:
    """
    创建归一化层
    
    Args:
        norm_type: 归一化类型 ('layernorm' 或 'rmsnorm')
        d_model: 模型维度
        eps: 防止除零的小值
        
    Returns:
        归一化层
    """
    if norm_type.lower() == 'layernorm':
        return LayerNorm(d_model, eps)
    elif norm_type.lower() == 'rmsnorm':
        return RMSNorm(d_model, eps)
    else:
        raise ValueError(f"未知的归一化类型: {norm_type}")


if __name__ == "__main__":
    # 测试归一化层
    batch_size, seq_len, d_model = 4, 10, 512
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 测试LayerNorm
    layernorm = LayerNorm(d_model)
    out = layernorm(x)
    print(f"LayerNorm输出形状: {out.shape}")
    print(f"LayerNorm输出均值: {out.mean().item():.6f}")
    print(f"LayerNorm输出标准差: {out.std().item():.6f}")
    
    # 测试RMSNorm
    rmsnorm = RMSNorm(d_model)
    out = rmsnorm(x)
    print(f"\nRMSNorm输出形状: {out.shape}")
    print(f"RMSNorm输出均值: {out.mean().item():.6f}")
    print(f"RMSNorm输出标准差: {out.std().item():.6f}")

