"""
RNN解码器模块
支持注意力机制，LSTM和GRU，两层单向网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import Attention, create_attention


class RNNDecoder(nn.Module):
    """RNN解码器（带注意力机制）"""
    
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 cell_type: str = 'lstm',
                 attention_type: str = 'dot',
                 encoder_hidden_dim: int = None,
                 padding_idx: int = 0):
        """
        初始化解码器
        
        Args:
            vocab_size: 词汇表大小
            embed_dim: 词嵌入维度
            hidden_dim: 隐藏层维度
            num_layers: 层数
            dropout: dropout概率
            cell_type: RNN类型 ('lstm' 或 'gru')
            attention_type: 注意力类型 ('dot', 'general', 'additive')
            encoder_hidden_dim: 编码器隐藏层维度（用于注意力，默认与hidden_dim相同）
            padding_idx: padding索引
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        
        if encoder_hidden_dim is None:
            encoder_hidden_dim = hidden_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        
        # 词嵌入层
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=padding_idx
        )
        
        # RNN层（输入是词嵌入 + 上下文向量）
        rnn_class = nn.LSTM if self.cell_type == 'lstm' else nn.GRU
        self.rnn = rnn_class(
            embed_dim + encoder_hidden_dim,  # 拼接嵌入和上下文
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 注意力机制
        self.attention = create_attention(
            attention_type=attention_type,
            query_dim=hidden_dim,
            key_dim=encoder_hidden_dim,
            hidden_dim=hidden_dim
        )
        
        # 输出层
        self.fc_out = nn.Linear(
            hidden_dim + encoder_hidden_dim + embed_dim,
            vocab_size
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                tgt: torch.Tensor,
                hidden: torch.Tensor,
                encoder_outputs: torch.Tensor,
                src_mask: torch.Tensor = None):
        """
        前向传播（单步解码）
        
        Args:
            tgt: 目标词索引 [batch_size, 1]
            hidden: 上一步的隐藏状态
                - LSTM: (h, c)
                - GRU: h
            encoder_outputs: 编码器输出 [batch_size, src_len, encoder_hidden_dim]
            src_mask: 源序列掩码 [batch_size, src_len]
            
        Returns:
            output: 输出分布 [batch_size, vocab_size]
            hidden: 当前隐藏状态
            attention_weights: 注意力权重 [batch_size, src_len]
        """
        # 词嵌入
        embedded = self.embedding(tgt)  # [batch_size, 1, embed_dim]
        embedded = self.dropout(embedded)
        
        # 获取当前隐藏状态作为query（取最后一层）
        if self.cell_type == 'lstm':
            query = hidden[0][-1]  # [batch_size, hidden_dim]
        else:
            query = hidden[-1]  # [batch_size, hidden_dim]
        
        # 计算注意力
        context, attention_weights = self.attention(
            query, encoder_outputs, encoder_outputs, src_mask
        )  # context: [batch_size, encoder_hidden_dim]
        
        # 拼接嵌入和上下文
        context = context.unsqueeze(1)  # [batch_size, 1, encoder_hidden_dim]
        rnn_input = torch.cat([embedded, context], dim=2)
        # rnn_input: [batch_size, 1, embed_dim + encoder_hidden_dim]
        
        # RNN解码
        rnn_output, hidden = self.rnn(rnn_input, hidden)
        # rnn_output: [batch_size, 1, hidden_dim]
        
        # 拼接RNN输出、上下文、嵌入
        output = torch.cat([
            rnn_output.squeeze(1),
            context.squeeze(1),
            embedded.squeeze(1)
        ], dim=1)  # [batch_size, hidden_dim + encoder_hidden_dim + embed_dim]
        
        # 输出层
        output = self.fc_out(output)  # [batch_size, vocab_size]
        
        return output, hidden, attention_weights
    
    def init_embeddings(self, pretrained_embeddings: torch.Tensor):
        """
        使用预训练词向量初始化嵌入层
        
        Args:
            pretrained_embeddings: 预训练词向量 [vocab_size, embed_dim]
        """
        self.embedding.weight.data.copy_(pretrained_embeddings)
        print("已加载预训练词向量")


if __name__ == "__main__":
    # 测试解码器
    vocab_size = 10000
    embed_dim = 256
    hidden_dim = 512
    encoder_hidden_dim = 512
    batch_size = 4
    src_len = 15
    
    decoder = RNNDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.3,
        cell_type='lstm',
        attention_type='dot',
        encoder_hidden_dim=encoder_hidden_dim
    )
    
    # 随机输入
    tgt = torch.randint(0, vocab_size, (batch_size, 1))
    encoder_outputs = torch.randn(batch_size, src_len, encoder_hidden_dim)
    src_mask = torch.ones(batch_size, src_len)
    
    # 初始隐藏状态
    h = torch.randn(2, batch_size, hidden_dim)
    c = torch.randn(2, batch_size, hidden_dim)
    hidden = (h, c)
    
    # 前向传播
    output, hidden, attn_weights = decoder(tgt, hidden, encoder_outputs, src_mask)
    
    print(f"解码器输出: {output.shape}")
    print(f"注意力权重: {attn_weights.shape}")

