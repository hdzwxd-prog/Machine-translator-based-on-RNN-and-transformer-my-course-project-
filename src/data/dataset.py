"""
数据集模块
功能：PyTorch数据集类，支持批处理和padding
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
from .vocab import Vocabulary


class TranslationDataset(Dataset):
    """翻译数据集类"""
    
    def __init__(self, 
                 src_token_lists: List[List[str]], 
                 tgt_token_lists: List[List[str]],
                 src_vocab: Vocabulary,
                 tgt_vocab: Vocabulary):
        """
        初始化数据集
        
        Args:
            src_token_lists: 源语言词元列表
            tgt_token_lists: 目标语言词元列表
            src_vocab: 源语言词汇表
            tgt_vocab: 目标语言词汇表
        """
        assert len(src_token_lists) == len(tgt_token_lists), \
            "源语言和目标语言数据长度不一致"
        
        self.src_token_lists = src_token_lists
        self.tgt_token_lists = tgt_token_lists
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def __len__(self) -> int:
        return len(self.src_token_lists)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            (源语言索引张量, 目标语言索引张量)
        """
        src_tokens = self.src_token_lists[idx]
        tgt_tokens = self.tgt_token_lists[idx]
        
        # 编码：源语言不加SOS/EOS，目标语言加SOS/EOS
        src_indices = self.src_vocab.encode(src_tokens, add_sos=False, add_eos=True)
        tgt_indices = self.tgt_vocab.encode(tgt_tokens, add_sos=True, add_eos=True)
        
        return torch.tensor(src_indices, dtype=torch.long), \
               torch.tensor(tgt_indices, dtype=torch.long)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    批处理整理函数：对序列进行padding
    
    Args:
        batch: 批次数据
        
    Returns:
        src_batch: 源语言批次 [batch_size, src_max_len]
        src_lengths: 源语言长度 [batch_size]
        tgt_batch: 目标语言批次 [batch_size, tgt_max_len]
        tgt_lengths: 目标语言长度 [batch_size]
    """
    src_batch, tgt_batch = zip(*batch)
    
    # 获取序列长度
    src_lengths = torch.tensor([len(seq) for seq in src_batch], dtype=torch.long)
    tgt_lengths = torch.tensor([len(seq) for seq in tgt_batch], dtype=torch.long)
    
    # Padding（padding_value=0，即PAD_IDX）
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    return src_batch, src_lengths, tgt_batch, tgt_lengths


def create_data_loader(dataset: TranslationDataset,
                       batch_size: int,
                       shuffle: bool = True,
                       num_workers: int = 0,
                       sampler=None,
                       pin_memory: bool = True) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱（如果提供了sampler则忽略）
        num_workers: 工作进程数
        sampler: 采样器（用于分布式训练）
        pin_memory: 是否固定内存
        
    Returns:
        DataLoader对象
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


if __name__ == "__main__":
    # 测试代码
    from .vocab import Vocabulary
    
    # 构建词汇表
    src_vocab = Vocabulary(min_freq=1)
    tgt_vocab = Vocabulary(min_freq=1)
    
    src_tokens = [['这', '是', '测试'], ['另一个', '测试', '句子']]
    tgt_tokens = [['this', 'is', 'test'], ['another', 'test', 'sentence']]
    
    src_vocab.build_vocab(src_tokens)
    tgt_vocab.build_vocab(tgt_tokens)
    
    # 创建数据集
    dataset = TranslationDataset(src_tokens, tgt_tokens, src_vocab, tgt_vocab)
    
    # 创建数据加载器
    dataloader = create_data_loader(dataset, batch_size=2, shuffle=False)
    
    for src_batch, src_lengths, tgt_batch, tgt_lengths in dataloader:
        print(f"源语言批次: {src_batch}")
        print(f"源语言长度: {src_lengths}")
        print(f"目标语言批次: {tgt_batch}")
        print(f"目标语言长度: {tgt_lengths}")

