"""
数据预处理模块
功能：数据清洗、分词处理
"""
import re
import jieba
import nltk
from typing import List, Tuple
import json


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, max_length: int = 100):
        """
        初始化预处理器
        
        Args:
            max_length: 最大句子长度
        """
        self.max_length = max_length
        # 初始化NLTK数据（避免训练时卡住）
        # 新版本NLTK使用punkt_tab，旧版本使用punkt
        # 如果数据不存在，抛出清晰的错误提示，而不是自动下载（避免卡住）
        punkt_found = False
        try:
            nltk.data.find('tokenizers/punkt_tab')  # 新版本NLTK (3.8+)
            punkt_found = True
        except LookupError:
            try:
                nltk.data.find('tokenizers/punkt')  # 旧版本NLTK
                punkt_found = True
            except LookupError:
                pass
        
        if not punkt_found:
            raise RuntimeError(
                "NLTK punkt数据未找到！\n"
                "请在训练前运行以下命令下载数据：\n"
                "  python setup_nltk.py\n"
                "或：\n"
                "  python -c \"import nltk; nltk.download('punkt_tab')\"\n"
                "  python -c \"import nltk; nltk.download('punkt')\"\n"
                "\n"
                "如果网络不可用，请确保已手动下载punkt数据到NLTK数据目录。"
            )
    
    def clean_text(self, text: str, lang: str = 'zh') -> str:
        """
        清洗文本：移除非法字符、多余空格等
        
        Args:
            text: 原始文本
            lang: 语言类型 ('zh' 或 'en')
            
        Returns:
            清洗后的文本
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 移除控制字符
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        if lang == 'zh':
            # 中文：保留中文字符、标点、数字、字母
            # 统一中英文标点
            text = text.replace('，', ',').replace('。', '.').replace('！', '!').replace('？', '?')
            text = text.replace('；', ';').replace('：', ':').replace('"', '"').replace('"', '"')
            text = text.replace(''', "'").replace(''', "'")
        else:
            # 英文：保留字母、数字、基本标点
            pass
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_chinese(self, text: str) -> List[str]:
        """
        中文分词（使用jieba）
        
        Args:
            text: 中文文本
            
        Returns:
            词元列表
        """
        tokens = jieba.lcut(text)
        # 过滤空白token
        tokens = [token.strip() for token in tokens if token.strip()]
        return tokens
    
    def tokenize_english(self, text: str) -> List[str]:
        """
        英文分词（基于空格和标点）
        
        Args:
            text: 英文文本
            
        Returns:
            词元列表
        """
        # 使用NLTK的word_tokenize
        tokens = nltk.word_tokenize(text.lower())
        return tokens
    
    def tokenize(self, text: str, lang: str) -> List[str]:
        """
        根据语言类型进行分词
        
        Args:
            text: 文本
            lang: 语言类型 ('zh' 或 'en')
            
        Returns:
            词元列表
        """
        if lang == 'zh':
            return self.tokenize_chinese(text)
        else:
            return self.tokenize_english(text)
    
    def is_valid_pair(self, src_tokens: List[str], tgt_tokens: List[str]) -> bool:
        """
        判断句对是否有效
        
        Args:
            src_tokens: 源语言词元
            tgt_tokens: 目标语言词元
            
        Returns:
            是否有效
        """
        # 检查长度
        if len(src_tokens) == 0 or len(tgt_tokens) == 0:
            return False
        if len(src_tokens) > self.max_length or len(tgt_tokens) > self.max_length:
            return False
        
        # 检查长度比例（避免翻译长度差异过大）
        ratio = len(src_tokens) / len(tgt_tokens)
        if ratio < 0.2 or ratio > 5.0:
            return False
        
        return True
    
    def process_pair(self, src_text: str, tgt_text: str, 
                     src_lang: str, tgt_lang: str) -> Tuple[List[str], List[str]]:
        """
        处理单个句对
        
        Args:
            src_text: 源语言文本
            tgt_text: 目标语言文本
            src_lang: 源语言类型
            tgt_lang: 目标语言类型
            
        Returns:
            (源语言词元列表, 目标语言词元列表) 或 (None, None)
        """
        # 清洗
        src_clean = self.clean_text(src_text, src_lang)
        tgt_clean = self.clean_text(tgt_text, tgt_lang)
        
        # 分词
        src_tokens = self.tokenize(src_clean, src_lang)
        tgt_tokens = self.tokenize(tgt_clean, tgt_lang)
        
        # 验证
        if not self.is_valid_pair(src_tokens, tgt_tokens):
            return None, None
        
        return src_tokens, tgt_tokens
    
    def load_and_process_data(self, filepath: str, src_lang: str, tgt_lang: str, 
                              max_samples: int = None) -> List[Tuple[List[str], List[str]]]:
        """
        加载并处理数据文件
        
        Args:
            filepath: 数据文件路径
            src_lang: 源语言类型
            tgt_lang: 目标语言类型
            max_samples: 最大样本数
            
        Returns:
            处理后的句对列表
        """
        pairs = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if max_samples and idx >= max_samples:
                    break
                
                try:
                    data = json.loads(line.strip())
                    src_text = data.get(src_lang, "")
                    tgt_text = data.get(tgt_lang, "")
                    
                    src_tokens, tgt_tokens = self.process_pair(
                        src_text, tgt_text, src_lang, tgt_lang
                    )
                    
                    if src_tokens is not None and tgt_tokens is not None:
                        pairs.append((src_tokens, tgt_tokens))
                    
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing line {idx}: {e}")
                    continue
        
        return pairs


if __name__ == "__main__":
    # 测试代码
    preprocessor = DataPreprocessor(max_length=100)
    
    # 测试中文分词
    zh_text = "这是一个测试句子，用于测试中文分词功能。"
    zh_tokens = preprocessor.tokenize(zh_text, 'zh')
    print(f"中文分词结果: {zh_tokens}")
    
    # 测试英文分词
    en_text = "This is a test sentence for English tokenization."
    en_tokens = preprocessor.tokenize(en_text, 'en')
    print(f"英文分词结果: {en_tokens}")
    
    # 测试句对处理
    src_tokens, tgt_tokens = preprocessor.process_pair(zh_text, en_text, 'zh', 'en')
    print(f"句对处理结果: {src_tokens} -> {tgt_tokens}")

