"""
评估指标模块
计算BLEU等机器翻译评估指标
"""
from typing import List
import sacrebleu


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """
    计算BLEU分数
    
    Args:
        predictions: 预测翻译列表
        references: 参考翻译列表
        
    Returns:
        BLEU分数
    """
    # sacrebleu需要references是列表的列表
    refs = [[ref] for ref in references]
    
    # 计算BLEU，使用force=True来suppress tokenized period警告
    # 因为我们的数据已经tokenized（用空格分隔），这是正常的
    bleu = sacrebleu.corpus_bleu(predictions, list(zip(*refs)), force=True)
    
    return bleu.score


def compute_bleu_sentence(prediction: str, reference: str) -> float:
    """
    计算单句BLEU分数
    
    Args:
        prediction: 预测翻译
        reference: 参考翻译
        
    Returns:
        BLEU分数
    """
    bleu = sacrebleu.sentence_bleu(prediction, [reference])
    return bleu.score


if __name__ == "__main__":
    # 测试BLEU计算
    preds = ["this is a test", "another test sentence"]
    refs = ["this is a test", "this is another test"]
    
    bleu_score = compute_bleu(preds, refs)
    print(f"BLEU Score: {bleu_score:.2f}")

