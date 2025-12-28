"""
多头注意力机制模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制（Multi-Head Attention）"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,
                 use_relative_pos: bool = False, max_len: int = 5000):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: Dropout比例
            use_relative_pos: 是否使用相对位置编码
            max_len: 最大序列长度（用于相对位置编码）
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_relative_pos = use_relative_pos
        
        # Q, K, V投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # 相对位置编码（如果使用）
        if use_relative_pos:
            self.relative_pos_emb = nn.Embedding(2 * max_len + 1, self.d_k)
        else:
            self.relative_pos_emb = None
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None, relative_pos_emb: torch.Tensor = None) -> tuple:
        """
        前向传播
        
        Args:
            query: [batch_size, q_len, d_model]
            key: [batch_size, k_len, d_model]
            value: [batch_size, v_len, d_model]
            mask: [batch_size, q_len, k_len] 或 [batch_size, 1, k_len]
            relative_pos_emb: [q_len, k_len, d_k] 相对位置编码（可选）
            
        Returns:
            output: [batch_size, q_len, d_model]
            attention_weights: [batch_size, num_heads, q_len, k_len]
        """
        batch_size = query.size(0)
        q_len = query.size(1)
        k_len = key.size(1)
        
        # 线性投影并重塑为多头
        Q = self.W_q(query).view(batch_size, q_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, k_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, k_len, self.num_heads, self.d_k).transpose(1, 2)
        # Q, K, V: [batch_size, num_heads, seq_len, d_k]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [batch_size, num_heads, q_len, k_len]
        
        # 添加相对位置编码（如果使用）
        if self.use_relative_pos and relative_pos_emb is not None:
            # relative_pos_emb: [q_len, k_len, d_k]
            # 需要转换为 [batch_size, num_heads, q_len, k_len]
            rel_pos_scores = torch.einsum('bhqd,qkd->bhqk', Q, relative_pos_emb)
            scores = scores + rel_pos_scores
        
        # 应用掩码
        if mask is not None:
            # mask: [batch_size, q_len, k_len] 或 [batch_size, 1, k_len]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch_size, 1, q_len, k_len]
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, k_len]
            # 使用-1e4而不是-1e9，避免FP16溢出
            scores = scores.masked_fill(mask == 0, -1e4)
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重到值
        output = torch.matmul(attention_weights, V)
        # output: [batch_size, num_heads, q_len, d_k]
        
        # 拼接多头
        output = output.transpose(1, 2).contiguous().view(
            batch_size, q_len, self.d_model
        )
        # output: [batch_size, q_len, d_model]
        
        # 输出投影
        output = self.W_o(output)
        
        return output, attention_weights


if __name__ == "__main__":
    # 测试多头注意力
    batch_size, seq_len, d_model = 4, 10, 512
    num_heads = 8
    
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, seq_len, seq_len)
    
    attn = MultiHeadAttention(d_model, num_heads)
    output, weights = attn(query, key, value, mask)
    
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {weights.shape}")

