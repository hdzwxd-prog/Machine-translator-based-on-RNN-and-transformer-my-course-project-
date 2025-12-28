"""
评估脚本
支持贪心解码和束搜索解码
"""
import os
import yaml
import torch
import argparse
from tqdm import tqdm

from src.data.preprocessor import DataPreprocessor
from src.data.vocab import Vocabulary
from src.models.rnn import RNNEncoder, RNNDecoder, Seq2Seq
from src.models.transformer import TransformerEncoder, TransformerDecoder, TransformerSeq2Seq
from src.decoding import GreedyDecoder, BeamSearchDecoder
from src.utils.metrics import compute_bleu


def load_config(config_path: str):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config, checkpoint_path, src_vocab, tgt_vocab, device):
    """加载模型（支持RNN和Transformer）"""
    model_config = config['model']
    model_type = model_config.get('type', 'rnn').lower()
    
    if model_type == 'rnn':
        # RNN模型
        # 创建编码器
        encoder = RNNEncoder(
            vocab_size=len(src_vocab),
            embed_dim=model_config['encoder']['embed_dim'],
            hidden_dim=model_config['encoder']['hidden_dim'],
            num_layers=model_config['encoder']['num_layers'],
            dropout=model_config['encoder']['dropout'],
            cell_type=model_config['cell_type'],
            bidirectional=model_config['encoder']['bidirectional'],
            padding_idx=src_vocab.PAD_IDX
        )
        
        # 编码器输出维度
        encoder_output_dim = model_config['encoder']['hidden_dim']
        if model_config['encoder']['bidirectional']:
            encoder_output_dim *= 2
        
        # 创建解码器
        decoder = RNNDecoder(
            vocab_size=len(tgt_vocab),
            embed_dim=model_config['decoder']['embed_dim'],
            hidden_dim=model_config['decoder']['hidden_dim'],
            num_layers=model_config['decoder']['num_layers'],
            dropout=model_config['decoder']['dropout'],
            cell_type=model_config['cell_type'],
            attention_type=model_config['attention']['type'],
            encoder_hidden_dim=encoder_output_dim,
            padding_idx=tgt_vocab.PAD_IDX
        )
        
        # 创建模型
        model = Seq2Seq(encoder, decoder).to(device)
        
    elif model_type == 'transformer':
        # Transformer模型
        encoder = TransformerEncoder(
            vocab_size=len(src_vocab),
            d_model=model_config['encoder']['d_model'],
            num_heads=model_config['encoder']['num_heads'],
            num_layers=model_config['encoder']['num_layers'],
            d_ff=model_config['encoder']['d_ff'],
            max_len=model_config['encoder']['max_len'],
            dropout=model_config['encoder']['dropout'],
            padding_idx=src_vocab.PAD_IDX,
            pos_encoding_type=model_config.get('pos_encoding_type', 'absolute'),
            norm_type=model_config.get('norm_type', 'layernorm'),
            use_relative_pos=model_config.get('use_relative_pos', False)
        )
        
        decoder = TransformerDecoder(
            vocab_size=len(tgt_vocab),
            d_model=model_config['decoder']['d_model'],
            num_heads=model_config['decoder']['num_heads'],
            num_layers=model_config['decoder']['num_layers'],
            d_ff=model_config['decoder']['d_ff'],
            max_len=model_config['decoder']['max_len'],
            dropout=model_config['decoder']['dropout'],
            padding_idx=tgt_vocab.PAD_IDX,
            pos_encoding_type=model_config.get('pos_encoding_type', 'absolute'),
            norm_type=model_config.get('norm_type', 'layernorm'),
            use_relative_pos=model_config.get('use_relative_pos', False)
        )
        
        model = TransformerSeq2Seq(encoder, decoder).to(device)
        
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型已从 {checkpoint_path} 加载")
    print(f"模型类型: {model_type}")
    print(f"Epoch: {checkpoint['epoch']}, Valid Loss: {checkpoint['best_valid_loss']:.4f}")
    
    return model


def translate(model, src_tokens, src_vocab, tgt_vocab, decoder_obj, device):
    """
    翻译单个句子
    
    Args:
        model: Seq2Seq模型
        src_tokens: 源语言词元列表
        src_vocab: 源语言词汇表
        tgt_vocab: 目标语言词汇表
        decoder_obj: 解码器对象（GreedyDecoder或BeamSearchDecoder）
        device: 设备
        
    Returns:
        翻译结果字符串
    """
    # 编码源句子
    src_indices = src_vocab.encode(src_tokens, add_sos=False, add_eos=True)
    src_tensor = torch.tensor([src_indices], dtype=torch.long).to(device)
    src_lengths = torch.tensor([len(src_indices)], dtype=torch.long).to(device)
    
    # 解码
    decoded_indices, _ = decoder_obj.decode(src_tensor, src_lengths)
    
    # 解码为词元
    decoded_tokens = tgt_vocab.decode(decoded_indices, skip_special=True)
    
    # 拼接为字符串
    translation = ' '.join(decoded_tokens)
    
    return translation


def evaluate_dataset(model, test_pairs, src_vocab, tgt_vocab, decoder_obj, device):
    """
    评估测试集
    
    Args:
        model: Seq2Seq模型
        test_pairs: 测试数据对列表
        src_vocab: 源语言词汇表
        tgt_vocab: 目标语言词汇表
        decoder_obj: 解码器对象
        device: 设备
        
    Returns:
        BLEU分数
    """
    predictions = []
    references = []
    
    print("开始翻译测试集...")
    for src_tokens, tgt_tokens in tqdm(test_pairs[:1000]):  # 限制评估样本数
        # 翻译
        translation = translate(model, src_tokens, src_vocab, tgt_vocab, decoder_obj, device)
        
        # 参考翻译
        reference = ' '.join(tgt_tokens)
        
        predictions.append(translation)
        references.append(reference)
    
    # 计算BLEU
    bleu_score = compute_bleu(predictions, references)
    
    return bleu_score, predictions, references


def main():
    parser = argparse.ArgumentParser(description='评估机器翻译模型')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt', help='模型检查点路径')
    parser.add_argument('--strategy', type=str, default='beam_search', choices=['greedy', 'beam_search'], help='解码策略')
    parser.add_argument('--beam_size', type=int, default=5, help='束搜索的束大小')
    parser.add_argument('--output', type=str, default='results.txt', help='输出文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备（支持指定GPU ID）
    device_str = config.get('device', 'cuda')
    if isinstance(device_str, str) and ':' in device_str:
        # 格式如 "cuda:1"
        device = torch.device(device_str)
    elif config.get('gpu_id') is not None:
        # 配置中指定了gpu_id
        gpu_id = config.get('gpu_id')
        device = torch.device(f'cuda:{gpu_id}')
    else:
        # 默认使用cuda或cpu
        device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    
    # 确保设置当前CUDA设备
    if device.type == 'cuda':
        gpu_id = device.index if device.index is not None else 0
        torch.cuda.set_device(gpu_id)
    
    print(f"使用设备: {device}")
    
    # 加载词汇表
    print("\n加载词汇表...")
    src_vocab = Vocabulary.load('vocabs/src_vocab.pkl')
    tgt_vocab = Vocabulary.load('vocabs/tgt_vocab.pkl')
    
    # 加载模型
    print("\n加载模型...")
    model = load_model(config, args.checkpoint, src_vocab, tgt_vocab, device)
    
    # 创建解码器
    print(f"\n创建解码器: {args.strategy}")
    if args.strategy == 'greedy':
        decoder_obj = GreedyDecoder(
            model=model,
            sos_idx=tgt_vocab.SOS_IDX,
            eos_idx=tgt_vocab.EOS_IDX,
            max_length=config['decoding']['max_length']
        )
    else:
        decoder_obj = BeamSearchDecoder(
            model=model,
            sos_idx=tgt_vocab.SOS_IDX,
            eos_idx=tgt_vocab.EOS_IDX,
            beam_size=args.beam_size,
            max_length=config['decoding']['max_length'],
            length_penalty=config['decoding']['length_penalty']
        )
    
    # 加载测试数据
    print("\n加载测试数据...")
    preprocessor = DataPreprocessor(max_length=config['data']['max_length'])
    test_file = os.path.join(config['data']['data_dir'], config['data']['test_file'])
    test_pairs = preprocessor.load_and_process_data(
        test_file,
        config['data']['source_lang'],
        config['data']['target_lang']
    )
    print(f"测试集样本数: {len(test_pairs)}")
    
    # 评估
    print("\n开始评估...")
    bleu_score, predictions, references = evaluate_dataset(
        model, test_pairs, src_vocab, tgt_vocab, decoder_obj, device
    )
    
    print(f"\nBLEU Score: {bleu_score:.2f}")
    
    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(f"解码策略: {args.strategy}\n")
        if args.strategy == 'beam_search':
            f.write(f"束大小: {args.beam_size}\n")
        f.write(f"BLEU Score: {bleu_score:.2f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("翻译示例\n")
        f.write("=" * 80 + "\n\n")
        
        for i in range(min(10, len(predictions))):
            f.write(f"样本 {i+1}:\n")
            f.write(f"源句子: {' '.join(test_pairs[i][0])}\n")
            f.write(f"参考翻译: {references[i]}\n")
            f.write(f"模型翻译: {predictions[i]}\n\n")
    
    print(f"\n结果已保存到 {args.output}")
    
    # 打印一些示例
    print("\n翻译示例:")
    print("=" * 80)
    for i in range(min(5, len(predictions))):
        print(f"\n样本 {i+1}:")
        print(f"源句子: {' '.join(test_pairs[i][0])}")
        print(f"参考翻译: {references[i]}")
        print(f"模型翻译: {predictions[i]}")


if __name__ == "__main__":
    main()

