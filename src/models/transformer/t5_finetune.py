"""
T5预训练模型微调模块
"""
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Optional


class T5FinetuneModel(nn.Module):
    """基于T5的微调模型"""
    
    def __init__(self, model_name: str = 't5-small', max_length: int = 512):
        """
        Args:
            model_name: T5模型名称 ('t5-small', 't5-base', 't5-large')
            max_length: 最大序列长度
        """
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        
        # 加载预训练T5模型和tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # 添加特殊token（如果需要）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor = None,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        """
        前向传播
        
        Args:
            src: 源序列token IDs [batch_size, src_len]
            tgt: 目标序列token IDs [batch_size, tgt_len]（训练时）
            src_mask: 源序列掩码（可选，T5会自动处理）
            tgt_mask: 目标序列掩码（可选）
            teacher_forcing_ratio: Teacher Forcing比例（T5总是使用teacher forcing）
            
        Returns:
            outputs: 输出logits [batch_size, tgt_len-1, vocab_size]
        """
        if tgt is not None:
            # 训练模式：使用teacher forcing
            # T5的输入格式：将任务前缀添加到源序列
            # 对于翻译任务，我们使用"translate Chinese to English: "前缀
            # 注意：这里简化处理，实际使用时应该在tokenization阶段添加前缀
            
            # 准备decoder输入（去掉最后一个token，因为T5需要shift right）
            decoder_input_ids = tgt[:, :-1].contiguous()
            labels = tgt[:, 1:].contiguous()
            
            # T5 forward
            outputs = self.model(
                input_ids=src,
                decoder_input_ids=decoder_input_ids,
                labels=labels
            )
            
            # 返回logits [batch_size, tgt_len-1, vocab_size]
            return outputs.logits
        else:
            # 推理模式
            raise NotImplementedError("推理模式需要单独实现generate方法")
    
    def generate(self, src: torch.Tensor, max_length: int = 100,
                 num_beams: int = 5, **kwargs) -> torch.Tensor:
        """
        生成翻译
        
        Args:
            src: 源序列token IDs [batch_size, src_len]
            max_length: 最大生成长度
            num_beams: Beam search大小
            **kwargs: 其他生成参数
            
        Returns:
            generated_ids: 生成的token IDs [batch_size, gen_len]
        """
        # T5生成
        generated_ids = self.model.generate(
            input_ids=src,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            **kwargs
        )
        
        return generated_ids
    
    def get_tokenizer(self):
        """获取tokenizer"""
        return self.tokenizer


def create_t5_model(model_name: str = 't5-small', max_length: int = 512):
    """
    创建T5微调模型
    
    Args:
        model_name: T5模型名称
        max_length: 最大序列长度
        
    Returns:
        T5模型和tokenizer
    """
    model = T5FinetuneModel(model_name, max_length)
    tokenizer = model.get_tokenizer()
    return model, tokenizer

