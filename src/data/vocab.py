"""
词汇表模块
功能：构建和管理词汇表，支持词频统计和低频词过滤
"""
from collections import Counter
from typing import List, Dict
import pickle


class Vocabulary:
    """词汇表类"""
    
    # 特殊标记
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    SOS_TOKEN = '<sos>'  # Start of Sequence
    EOS_TOKEN = '<eos>'  # End of Sequence
    
    PAD_IDX = 0
    UNK_IDX = 1
    SOS_IDX = 2
    EOS_IDX = 3
    
    def __init__(self, min_freq: int = 2, max_size: int = 50000):
        """
        初始化词汇表
        
        Args:
            min_freq: 最小词频阈值
            max_size: 最大词汇表大小
        """
        self.min_freq = min_freq
        self.max_size = max_size
        
        # 词到索引的映射
        self.word2idx: Dict[str, int] = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
            self.SOS_TOKEN: self.SOS_IDX,
            self.EOS_TOKEN: self.EOS_IDX,
        }
        
        # 索引到词的映射
        self.idx2word: Dict[int, str] = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.UNK_IDX: self.UNK_TOKEN,
            self.SOS_IDX: self.SOS_TOKEN,
            self.EOS_IDX: self.EOS_TOKEN,
        }
        
        # 词频统计
        self.word_freq: Counter = Counter()
    
    def build_vocab(self, token_lists: List[List[str]]) -> None:
        """
        从词元列表构建词汇表
        
        Args:
            token_lists: 词元列表的列表
        """
        # 统计词频
        for tokens in token_lists:
            self.word_freq.update(tokens)
        
        # 按词频排序并过滤低频词
        sorted_words = [
            word for word, freq in self.word_freq.most_common()
            if freq >= self.min_freq
        ]
        
        # 限制词汇表大小
        if len(sorted_words) > self.max_size - 4:  # 减去4个特殊标记
            sorted_words = sorted_words[:self.max_size - 4]
        
        # 构建映射
        for word in sorted_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        print(f"词汇表构建完成，共 {len(self.word2idx)} 个词")
        print(f"过滤掉的低频词: {len(self.word_freq) - len(self.word2idx) + 4}")
    
    def encode(self, tokens: List[str], add_sos: bool = False, add_eos: bool = False) -> List[int]:
        """
        将词元序列编码为索引序列
        
        Args:
            tokens: 词元列表
            add_sos: 是否添加起始标记
            add_eos: 是否添加结束标记
            
        Returns:
            索引列表
        """
        indices = []
        
        if add_sos:
            indices.append(self.SOS_IDX)
        
        for token in tokens:
            indices.append(self.word2idx.get(token, self.UNK_IDX))
        
        if add_eos:
            indices.append(self.EOS_IDX)
        
        return indices
    
    def decode(self, indices: List[int], skip_special: bool = True) -> List[str]:
        """
        将索引序列解码为词元序列
        
        Args:
            indices: 索引列表
            skip_special: 是否跳过特殊标记
            
        Returns:
            词元列表
        """
        tokens = []
        special_indices = {self.PAD_IDX, self.SOS_IDX, self.EOS_IDX}
        
        for idx in indices:
            if skip_special and idx in special_indices:
                continue
            if idx == self.EOS_IDX and skip_special:
                break
            tokens.append(self.idx2word.get(idx, self.UNK_TOKEN))
        
        return tokens
    
    def __len__(self) -> int:
        """返回词汇表大小"""
        return len(self.word2idx)
    
    def save(self, filepath: str) -> None:
        """
        保存词汇表到文件
        
        Args:
            filepath: 保存路径
        """
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': self.word_freq,
            'min_freq': self.min_freq,
            'max_size': self.max_size,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"词汇表已保存到 {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        """
        从文件加载词汇表
        
        Args:
            filepath: 文件路径
            
        Returns:
            Vocabulary对象
        """
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        vocab = cls(min_freq=vocab_data['min_freq'], max_size=vocab_data['max_size'])
        vocab.word2idx = vocab_data['word2idx']
        vocab.idx2word = vocab_data['idx2word']
        vocab.word_freq = vocab_data['word_freq']
        
        print(f"词汇表已从 {filepath} 加载，共 {len(vocab)} 个词")
        return vocab


if __name__ == "__main__":
    # 测试代码
    token_lists = [
        ['这', '是', '一个', '测试'],
        ['测试', '词汇表', '构建'],
        ['词汇表', '词频', '统计'],
    ]
    
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocab(token_lists)
    
    # 测试编码
    tokens = ['这', '是', '测试', '未知词']
    indices = vocab.encode(tokens, add_sos=True, add_eos=True)
    print(f"编码结果: {tokens} -> {indices}")
    
    # 测试解码
    decoded = vocab.decode(indices)
    print(f"解码结果: {indices} -> {decoded}")

