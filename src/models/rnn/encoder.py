"""
RNN编码器模块
支持LSTM和GRU，两层单向网络
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNEncoder(nn.Module):
    """RNN编码器"""
    
    def __init__(self, 
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 cell_type: str = 'lstm',
                 bidirectional: bool = False,
                 padding_idx: int = 0):
        """
        初始化编码器
        
        Args:
            vocab_size: 词汇表大小
            embed_dim: 词嵌入维度
            hidden_dim: 隐藏层维度
            num_layers: 层数
            dropout: dropout概率
            cell_type: RNN类型 ('lstm' 或 'gru')
            bidirectional: 是否双向
            padding_idx: padding索引
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # 词嵌入层
        self.embedding = nn.Embedding(
            vocab_size, 
            embed_dim, 
            padding_idx=padding_idx
        )
        
        # RNN层
        rnn_class = nn.LSTM if self.cell_type == 'lstm' else nn.GRU
        self.rnn = rnn_class(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor):
        """
        前向传播
        
        Args:
            src: 源序列 [batch_size, seq_len]
            src_lengths: 序列长度 [batch_size]
            
        Returns:
            outputs: 编码器输出 [batch_size, seq_len, hidden_dim * num_directions]
            hidden: 最终隐藏状态
                - LSTM: (h_n, c_n)
                    h_n: [num_layers * num_directions, batch_size, hidden_dim]
                    c_n: [num_layers * num_directions, batch_size, hidden_dim]
                - GRU: h_n [num_layers * num_directions, batch_size, hidden_dim]
        """
        # 词嵌入
        embedded = self.embedding(src)  # [batch_size, seq_len, embed_dim]
        embedded = self.dropout(embedded)
        
        # Pack序列（提高效率）
        packed = pack_padded_sequence(
            embedded, 
            src_lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # RNN编码
        packed_outputs, hidden = self.rnn(packed)
        
        # Unpack序列
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        # outputs: [batch_size, seq_len, hidden_dim * num_directions]
        
        return outputs, hidden
    
    def init_embeddings(self, pretrained_embeddings: torch.Tensor):
        """
        使用预训练词向量初始化嵌入层
        
        Args:
            pretrained_embeddings: 预训练词向量 [vocab_size, embed_dim]
        """
        self.embedding.weight.data.copy_(pretrained_embeddings)
        print("已加载预训练词向量")


if __name__ == "__main__":
    # 测试编码器
    vocab_size = 10000
    embed_dim = 256
    hidden_dim = 512
    batch_size = 4
    seq_len = 15
    
    encoder = RNNEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.3,
        cell_type='lstm'
    )
    
    # 随机输入
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    src_lengths = torch.randint(5, seq_len+1, (batch_size,))
    
    # 前向传播
    outputs, hidden = encoder(src, src_lengths)
    
    print(f"编码器输出: {outputs.shape}")
    if isinstance(hidden, tuple):  # LSTM
        print(f"隐藏状态 h: {hidden[0].shape}, c: {hidden[1].shape}")
    else:  # GRU
        print(f"隐藏状态: {hidden.shape}")

