"""
注意力机制模块
实现多种注意力对齐函数：点积型、乘法型、加法型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """注意力机制基类"""
    
    def forward(self, query: torch.Tensor, keys: torch.Tensor, 
                values: torch.Tensor, mask: torch.Tensor = None):
        """
        计算注意力
        
        Args:
            query: 查询向量 [batch_size, query_dim]
            keys: 键向量 [batch_size, seq_len, key_dim]
            values: 值向量 [batch_size, seq_len, value_dim]
            mask: 掩码 [batch_size, seq_len]
            
        Returns:
            context: 上下文向量 [batch_size, value_dim]
            attention_weights: 注意力权重 [batch_size, seq_len]
        """
        raise NotImplementedError


class DotAttention(Attention):
    """点积注意力（Dot-Product Attention）
    
    score(query, key) = query · key
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
    
    def forward(self, query: torch.Tensor, keys: torch.Tensor,
                values: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            query: [batch_size, hidden_dim]
            keys: [batch_size, seq_len, hidden_dim]
            values: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, seq_len]
        """
        # 计算注意力分数
        # query: [batch_size, hidden_dim] -> [batch_size, 1, hidden_dim]
        # keys: [batch_size, seq_len, hidden_dim]
        # scores: [batch_size, 1, seq_len]
        scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))
        scores = scores.squeeze(1)  # [batch_size, seq_len]
        
        # 缩放
        scores = scores / (self.hidden_dim ** 0.5)
        
        # 应用掩码
        # 注意：使用-1e4而不是-1e9，避免混合精度训练（FP16）时的溢出错误
        # FP16的最大值约为65504，-1e9会导致"cannot be converted to type at::Half without overflow"
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len]
        
        # 计算上下文向量
        # attention_weights: [batch_size, 1, seq_len]
        # values: [batch_size, seq_len, hidden_dim]
        # context: [batch_size, 1, hidden_dim] -> [batch_size, hidden_dim]
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)
        
        return context, attention_weights


class GeneralAttention(Attention):
    """乘法注意力（General/Multiplicative Attention）
    
    score(query, key) = query · W · key
    """
    
    def __init__(self, query_dim: int, key_dim: int):
        super().__init__()
        self.W = nn.Linear(key_dim, query_dim, bias=False)
    
    def forward(self, query: torch.Tensor, keys: torch.Tensor,
                values: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            query: [batch_size, query_dim]
            keys: [batch_size, seq_len, key_dim]
            values: [batch_size, seq_len, value_dim]
            mask: [batch_size, seq_len]
        """
        # 转换keys
        transformed_keys = self.W(keys)  # [batch_size, seq_len, query_dim]
        
        # 计算注意力分数
        scores = torch.bmm(query.unsqueeze(1), transformed_keys.transpose(1, 2))
        scores = scores.squeeze(1)  # [batch_size, seq_len]
        
        # 应用掩码
        # 注意：使用-1e4而不是-1e9，避免混合精度训练（FP16）时的溢出错误
        # FP16的最大值约为65504，-1e9会导致"cannot be converted to type at::Half without overflow"
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        
        # 计算上下文向量
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)
        
        return context, attention_weights


class AdditiveAttention(Attention):
    """加法注意力（Additive/Concat Attention，也称Bahdanau Attention）
    
    score(query, key) = v^T · tanh(W1·query + W2·key)
    """
    
    def __init__(self, query_dim: int, key_dim: int, hidden_dim: int):
        super().__init__()
        self.W_query = nn.Linear(query_dim, hidden_dim, bias=False)
        self.W_key = nn.Linear(key_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, query: torch.Tensor, keys: torch.Tensor,
                values: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            query: [batch_size, query_dim]
            keys: [batch_size, seq_len, key_dim]
            values: [batch_size, seq_len, value_dim]
            mask: [batch_size, seq_len]
        """
        batch_size, seq_len, _ = keys.size()
        
        # 转换query和keys
        # query: [batch_size, hidden_dim] -> [batch_size, 1, hidden_dim]
        query_transformed = self.W_query(query).unsqueeze(1)
        # keys: [batch_size, seq_len, hidden_dim]
        keys_transformed = self.W_key(keys)
        
        # 加法 + tanh
        # [batch_size, seq_len, hidden_dim]
        combined = torch.tanh(query_transformed + keys_transformed)
        
        # 计算分数
        # [batch_size, seq_len, 1] -> [batch_size, seq_len]
        scores = self.v(combined).squeeze(-1)
        
        # 应用掩码
        # 注意：使用-1e4而不是-1e9，避免混合精度训练（FP16）时的溢出错误
        # FP16的最大值约为65504，-1e9会导致"cannot be converted to type at::Half without overflow"
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        
        # 计算上下文向量
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)
        
        return context, attention_weights


def create_attention(attention_type: str, query_dim: int, 
                     key_dim: int, hidden_dim: int) -> Attention:
    """
    创建注意力机制
    
    Args:
        attention_type: 注意力类型 ('dot', 'general', 'additive')
        query_dim: 查询维度
        key_dim: 键维度
        hidden_dim: 隐藏维度
        
    Returns:
        Attention对象
    """
    if attention_type == 'dot':
        assert query_dim == key_dim, "点积注意力要求query_dim == key_dim"
        return DotAttention(hidden_dim=query_dim)
    elif attention_type == 'general':
        return GeneralAttention(query_dim, key_dim)
    elif attention_type == 'additive':
        return AdditiveAttention(query_dim, key_dim, hidden_dim)
    else:
        raise ValueError(f"未知的注意力类型: {attention_type}")


if __name__ == "__main__":
    # 测试注意力机制
    batch_size, seq_len, hidden_dim = 4, 10, 512
    
    query = torch.randn(batch_size, hidden_dim)
    keys = torch.randn(batch_size, seq_len, hidden_dim)
    values = torch.randn(batch_size, seq_len, hidden_dim)
    mask = torch.ones(batch_size, seq_len)
    
    # 测试点积注意力
    dot_attn = DotAttention(hidden_dim)
    context, weights = dot_attn(query, keys, values, mask)
    print(f"点积注意力 - Context: {context.shape}, Weights: {weights.shape}")
    
    # 测试乘法注意力
    general_attn = GeneralAttention(hidden_dim, hidden_dim)
    context, weights = general_attn(query, keys, values, mask)
    print(f"乘法注意力 - Context: {context.shape}, Weights: {weights.shape}")
    
    # 测试加法注意力
    additive_attn = AdditiveAttention(hidden_dim, hidden_dim, hidden_dim)
    context, weights = additive_attn(query, keys, values, mask)
    print(f"加法注意力 - Context: {context.shape}, Weights: {weights.shape}")

