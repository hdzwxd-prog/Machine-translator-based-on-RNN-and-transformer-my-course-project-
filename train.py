"""
主训练脚本
支持分布式训练（DDP）和混合精度训练（AMP）
"""
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import random
import numpy as np
from pathlib import Path
import hashlib
import pickle

from src.data.preprocessor import DataPreprocessor
from src.data.vocab import Vocabulary
from src.data.dataset import TranslationDataset, create_data_loader
from src.models.rnn import RNNEncoder, RNNDecoder, Seq2Seq
from src.models.transformer import TransformerEncoder, TransformerDecoder, TransformerSeq2Seq
# T5模型延迟导入（仅在需要时）
# from src.models.transformer.t5_finetune import T5FinetuneModel
from src.training.trainer import Trainer


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """
    初始化分布式训练环境
    
    注意：如果使用torchrun启动，分布式环境已经自动初始化，
         不需要调用此函数，否则会导致端口冲突和卡死。
    
    Args:
        rank: 当前进程的rank
        world_size: 总进程数（GPU数量）
        backend: 分布式后端
    """
    # 检查是否已经初始化（torchrun会自动初始化）
    if dist.is_initialized():
        print(f"[WARNING] 分布式环境已初始化，跳过setup_distributed")
        return
    
    # 使用环境变量中的端口（如果torchrun设置了）
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '12355')
    
    print(f"[DEBUG] 初始化分布式环境: rank={rank}, world_size={world_size}")
    print(f"[DEBUG] MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    print(f"[DEBUG] 分布式环境初始化完成")


def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()


def load_config(config_path: str):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def compute_cache_key(data_config):
    """
    计算缓存键，基于数据配置参数
    
    Args:
        data_config: 数据配置字典
        
    Returns:
        cache_key: 缓存键字符串
    """
    # 提取影响预处理的关键参数
    key_params = {
        'data_dir': data_config['data_dir'],
        'train_file': data_config['train_file'],
        'valid_file': data_config['valid_file'],
        'source_lang': data_config['source_lang'],
        'target_lang': data_config['target_lang'],
        'max_length': data_config['max_length'],
        'min_freq': data_config['min_freq'],
        'max_vocab_size': data_config['max_vocab_size'],
    }
    
    # 将参数字典转换为字符串并计算hash
    key_str = json.dumps(key_params, sort_keys=True)
    cache_key = hashlib.md5(key_str.encode()).hexdigest()
    return cache_key


def check_cache_exists(cache_dir, cache_key):
    """
    检查缓存是否存在
    
    Args:
        cache_dir: 缓存目录
        cache_key: 缓存键
        
    Returns:
        bool: 缓存是否存在
    """
    cache_file = os.path.join(cache_dir, f'{cache_key}.pkl')
    return os.path.exists(cache_file)


def load_cached_data(cache_dir, cache_key):
    """
    从缓存加载预处理数据
    
    Args:
        cache_dir: 缓存目录
        cache_key: 缓存键
        
    Returns:
        tuple: (train_src, train_tgt, valid_src, valid_tgt, src_vocab, tgt_vocab)
    """
    cache_file = os.path.join(cache_dir, f'{cache_key}.pkl')
    print(f"\n从缓存加载预处理数据: {cache_file}")
    
    with open(cache_file, 'rb') as f:
        cached_data = pickle.load(f)
    
    train_src = cached_data['train_src']
    train_tgt = cached_data['train_tgt']
    valid_src = cached_data['valid_src']
    valid_tgt = cached_data['valid_tgt']
    
    # Vocabulary对象需要从字典重建
    src_vocab = Vocabulary(
        min_freq=cached_data['src_vocab']['min_freq'],
        max_size=cached_data['src_vocab']['max_size']
    )
    src_vocab.word2idx = cached_data['src_vocab']['word2idx']
    src_vocab.idx2word = cached_data['src_vocab']['idx2word']
    src_vocab.word_freq = cached_data['src_vocab']['word_freq']
    
    tgt_vocab = Vocabulary(
        min_freq=cached_data['tgt_vocab']['min_freq'],
        max_size=cached_data['tgt_vocab']['max_size']
    )
    tgt_vocab.word2idx = cached_data['tgt_vocab']['word2idx']
    tgt_vocab.idx2word = cached_data['tgt_vocab']['idx2word']
    tgt_vocab.word_freq = cached_data['tgt_vocab']['word_freq']
    
    print(f"缓存加载成功:")
    print(f"  训练集: {len(train_src)} 个样本")
    print(f"  验证集: {len(valid_src)} 个样本")
    print(f"  源语言词汇表: {len(src_vocab)} 个词")
    print(f"  目标语言词汇表: {len(tgt_vocab)} 个词")
    
    return train_src, train_tgt, valid_src, valid_tgt, src_vocab, tgt_vocab


def save_cached_data(cache_dir, cache_key, train_src, train_tgt, valid_src, valid_tgt, src_vocab, tgt_vocab):
    """
    保存预处理数据到缓存
    
    Args:
        cache_dir: 缓存目录
        cache_key: 缓存键
        train_src: 训练集源语言数据
        train_tgt: 训练集目标语言数据
        valid_src: 验证集源语言数据
        valid_tgt: 验证集目标语言数据
        src_vocab: 源语言词汇表
        tgt_vocab: 目标语言词汇表
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{cache_key}.pkl')
    
    # Vocabulary对象需要转换为字典以便序列化
    cached_data = {
        'train_src': train_src,
        'train_tgt': train_tgt,
        'valid_src': valid_src,
        'valid_tgt': valid_tgt,
        'src_vocab': {
            'word2idx': src_vocab.word2idx,
            'idx2word': src_vocab.idx2word,
            'word_freq': src_vocab.word_freq,
            'min_freq': src_vocab.min_freq,
            'max_size': src_vocab.max_size,
        },
        'tgt_vocab': {
            'word2idx': tgt_vocab.word2idx,
            'idx2word': tgt_vocab.idx2word,
            'word_freq': tgt_vocab.word_freq,
            'min_freq': tgt_vocab.min_freq,
            'max_size': tgt_vocab.max_size,
        },
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)
    
    print(f"\n预处理数据已保存到缓存: {cache_file}")


import json  # 添加json导入


def prepare_data(config, use_cache=True):
    """
    准备数据（支持缓存）
    
    Args:
        config: 配置字典
        use_cache: 是否使用缓存（默认True）
        
    Returns:
        train_loader, valid_loader, src_vocab, tgt_vocab
    """
    print("=" * 80)
    print("数据准备阶段")
    print("=" * 80)
    
    data_config = config['data']
    data_dir = data_config['data_dir']
    
    # 缓存目录和缓存键
    cache_dir = 'cache'
    cache_key = compute_cache_key(data_config)
    
    # 检查缓存
    if use_cache and check_cache_exists(cache_dir, cache_key):
        print(f"\n发现预处理缓存 (key: {cache_key[:8]}...)")
        try:
            train_src, train_tgt, valid_src, valid_tgt, src_vocab, tgt_vocab = load_cached_data(
                cache_dir, cache_key
            )
            # 直接跳到创建数据集
            skip_to_dataset = True
        except Exception as e:
            print(f"警告: 加载缓存失败 ({e})，将重新预处理")
            skip_to_dataset = False
    else:
        skip_to_dataset = False
        if use_cache:
            print(f"\n未找到预处理缓存，开始预处理 (key: {cache_key[:8]}...)")
        else:
            print(f"\n缓存已禁用，开始预处理")
    
    if not skip_to_dataset:
        # 初始化预处理器
        preprocessor = DataPreprocessor(max_length=data_config['max_length'])
        
        # 加载和处理训练数据
        print("\n加载训练数据...")
        train_file = os.path.join(data_dir, data_config['train_file'])
        train_pairs = preprocessor.load_and_process_data(
            train_file,
            data_config['source_lang'],
            data_config['target_lang']
        )
        print(f"训练集样本数: {len(train_pairs)}")
        
        # 加载和处理验证数据
        print("\n加载验证数据...")
        valid_file = os.path.join(data_dir, data_config['valid_file'])
        valid_pairs = preprocessor.load_and_process_data(
            valid_file,
            data_config['source_lang'],
            data_config['target_lang']
        )
        print(f"验证集样本数: {len(valid_pairs)}")
        
        # 分离源语言和目标语言
        train_src = [pair[0] for pair in train_pairs]
        train_tgt = [pair[1] for pair in train_pairs]
        valid_src = [pair[0] for pair in valid_pairs]
        valid_tgt = [pair[1] for pair in valid_pairs]
        
        # 构建词汇表
        print("\n构建词汇表...")
        src_vocab = Vocabulary(
            min_freq=data_config['min_freq'],
            max_size=data_config['max_vocab_size']
        )
        tgt_vocab = Vocabulary(
            min_freq=data_config['min_freq'],
            max_size=data_config['max_vocab_size']
        )
        
        src_vocab.build_vocab(train_src)
        tgt_vocab.build_vocab(train_tgt)
        
        # 保存词汇表（保持向后兼容）
        vocab_dir = 'vocabs'
        os.makedirs(vocab_dir, exist_ok=True)
        src_vocab.save(os.path.join(vocab_dir, 'src_vocab.pkl'))
        tgt_vocab.save(os.path.join(vocab_dir, 'tgt_vocab.pkl'))
        
        # 保存到缓存
        if use_cache:
            save_cached_data(
                cache_dir, cache_key,
                train_src, train_tgt, valid_src, valid_tgt,
                src_vocab, tgt_vocab
            )
    
    # 创建数据集
    print("\n创建数据集...")
    train_dataset = TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab)
    valid_dataset = TranslationDataset(valid_src, valid_tgt, src_vocab, tgt_vocab)
    
    # 创建数据加载器
    use_ddp = config['training'].get('use_ddp', False)
    num_workers = config['training'].get('num_workers', 0)
    pin_memory = config['training'].get('pin_memory', False)
    
    if use_ddp and dist.is_initialized():
        # 分布式训练：使用DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True
        )
        valid_sampler = DistributedSampler(
            valid_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False
        )
        
        train_loader = create_data_loader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,  # DistributedSampler已经处理shuffle
            num_workers=num_workers,
            sampler=train_sampler,
            pin_memory=pin_memory
        )
        valid_loader = create_data_loader(
            valid_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            sampler=valid_sampler,
            pin_memory=pin_memory
        )
    else:
        # 单GPU训练
        train_loader = create_data_loader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        valid_loader = create_data_loader(
            valid_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return train_loader, valid_loader, src_vocab, tgt_vocab


def build_model(config, src_vocab, tgt_vocab, device):
    """构建模型"""
    print("\n" + "=" * 80)
    print("模型构建阶段")
    print("=" * 80)
    
    model_config = config['model']
    model_type = model_config.get('type', 'rnn').lower()
    
    if model_type == 'rnn':
        # 创建RNN编码器
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
        
        # 创建RNN解码器
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
        
        # 创建RNN Seq2Seq模型
        model = Seq2Seq(encoder, decoder).to(device)
        
        # 打印模型信息
        print(f"\n模型类型: RNN")
        print(f"  编码器: {model_config['cell_type'].upper()}, "
              f"{model_config['encoder']['num_layers']}层, "
              f"隐藏维度={model_config['encoder']['hidden_dim']}")
        print(f"  解码器: {model_config['cell_type'].upper()}, "
              f"{model_config['decoder']['num_layers']}层, "
              f"隐藏维度={model_config['decoder']['hidden_dim']}")
        print(f"  注意力机制: {model_config['attention']['type']}")
    
    elif model_type == 'transformer':
        # 创建Transformer编码器
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
        
        # 创建Transformer解码器
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
        
        # 创建Transformer Seq2Seq模型
        model = TransformerSeq2Seq(encoder, decoder).to(device)
        
        # 打印模型信息
        print(f"\n模型类型: Transformer")
        print(f"  编码器: {model_config['encoder']['num_layers']}层, "
              f"d_model={model_config['encoder']['d_model']}, "
              f"num_heads={model_config['encoder']['num_heads']}, "
              f"d_ff={model_config['encoder']['d_ff']}")
        print(f"  解码器: {model_config['decoder']['num_layers']}层, "
              f"d_model={model_config['decoder']['d_model']}, "
              f"num_heads={model_config['decoder']['num_heads']}, "
              f"d_ff={model_config['decoder']['d_ff']}")
        print(f"  位置编码: {model_config.get('pos_encoding_type', 'absolute')}")
        print(f"  归一化方法: {model_config.get('norm_type', 'layernorm')}")
    
    elif model_type == 't5':
        # 延迟导入T5模型（仅在需要时）
        try:
            from src.models.transformer.t5_finetune import T5FinetuneModel
        except ImportError:
            raise ImportError("T5模型需要安装transformers库: pip install transformers")
        
        # 创建T5模型
        model = T5FinetuneModel(
            model_name=model_config.get('model_name', 't5-small'),
            max_length=model_config.get('max_length', 512)
        ).to(device)
        
        # 打印模型信息
        print(f"\n模型类型: T5 (预训练微调)")
        print(f"  预训练模型: {model_config.get('model_name', 't5-small')}")
        print(f"  最大长度: {model_config.get('max_length', 512)}")
        print(f"  注意: T5使用自己的tokenizer，不使用项目词汇表")
    
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    return model


def main(rank: int = 0, world_size: int = 1, config_path: str = 'config/config.yaml'):
    """
    主函数
    
    Args:
        rank: 当前进程rank（分布式训练时使用）
        world_size: 总进程数（分布式训练时使用）
        config_path: 配置文件路径
    """
    # 加载配置
    config = load_config(config_path)
    
    # 分布式训练设置
    use_ddp = config['training'].get('use_ddp', False) and torch.cuda.is_available()
    
    # 调试信息：检查分布式环境状态
    print(f"[DEBUG] use_ddp={use_ddp}, world_size={world_size}, rank={rank}")
    print(f"[DEBUG] dist.is_initialized()={dist.is_initialized()}")
    print(f"[DEBUG] LOCAL_RANK={os.environ.get('LOCAL_RANK', 'None')}")
    print(f"[DEBUG] WORLD_SIZE={os.environ.get('WORLD_SIZE', 'None')}")
    print(f"[DEBUG] MASTER_ADDR={os.environ.get('MASTER_ADDR', 'None')}")
    print(f"[DEBUG] MASTER_PORT={os.environ.get('MASTER_PORT', 'None')}")
    
    # 检查是否已经初始化（torchrun会自动初始化）
    if use_ddp and world_size > 1:
        if dist.is_initialized():
            # torchrun已经初始化，使用环境变量中的rank
            rank = int(os.environ.get('LOCAL_RANK', rank))
            world_size = int(os.environ.get('WORLD_SIZE', world_size))
            print(f"[DEBUG] 使用torchrun初始化的分布式环境: rank={rank}, world_size={world_size}")
        else:
            # 手动初始化（不推荐，仅用于非torchrun场景）
            print(f"[DEBUG] 手动初始化分布式环境")
            setup_distributed(rank, world_size, config.get('dist_backend', 'nccl'))
        
        device = torch.device(f'cuda:{rank}')
        is_main_process = (rank == 0)
        print(f"[DEBUG] 设备: {device}, 是否主进程: {is_main_process}")
    else:
        # 单GPU训练：检查是否指定了GPU ID
        if torch.cuda.is_available():
            # 检查环境变量CUDA_VISIBLE_DEVICES
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if cuda_visible:
                # 如果设置了CUDA_VISIBLE_DEVICES，使用cuda:0（因为环境变量会重新映射）
                device = torch.device('cuda:0')
                print(f"[DEBUG] 检测到CUDA_VISIBLE_DEVICES={cuda_visible}，使用cuda:0（映射到物理GPU {cuda_visible}）")
            else:
                # 检查配置中是否指定了GPU ID
                device_str = config.get('device', 'cuda')
                if isinstance(device_str, str) and ':' in device_str:
                    # 格式如 "cuda:1"
                    device = torch.device(device_str)
                elif config.get('gpu_id') is not None:
                    # 配置中指定了gpu_id
                    gpu_id = config.get('gpu_id')
                    device = torch.device(f'cuda:{gpu_id}')
                    print(f"[DEBUG] 配置中指定了gpu_id={gpu_id}")
                else:
                    # 默认使用cuda:0
                    device = torch.device('cuda:0')
                    print(f"[DEBUG] 未指定GPU ID，默认使用cuda:0")
        else:
            device = torch.device('cpu')
        
        # 确保设置当前CUDA设备（重要！）
        if device.type == 'cuda':
            gpu_id = device.index if device.index is not None else 0
            torch.cuda.set_device(gpu_id)
            print(f"[DEBUG] 设置CUDA当前设备为: {torch.cuda.current_device()}")
            print(f"[DEBUG] 验证设备: 实际使用GPU {torch.cuda.current_device()}")
        
        is_main_process = True
        rank = 0
        print(f"[DEBUG] 单GPU模式: {device}")
    
    # 设置随机种子
    set_seed(config['seed'] + rank)  # 不同进程使用不同种子
    
    if is_main_process:
        print(f"\n{'='*80}")
        print(f"训练配置")
        print(f"{'='*80}")
        print(f"使用设备: {device}")
        if use_ddp:
            print(f"分布式训练: {world_size} GPUs")
        print(f"混合精度训练: {config['training'].get('use_amp', False)}")
        print(f"批次大小: {config['training']['batch_size']}")
        print(f"训练轮数: {config['training']['num_epochs']}")
        print(f"{'='*80}\n")
    
    # 准备数据（添加调试信息）
    print(f"[DEBUG] 开始准备数据...")
    use_cache = config['training'].get('use_data_cache', True)
    train_loader, valid_loader, src_vocab, tgt_vocab = prepare_data(config, use_cache=use_cache)
    print(f"[DEBUG] 数据准备完成")
    
    # 再次确认设备设置（确保模型构建在正确的GPU上）
    if device.type == 'cuda':
        gpu_id = device.index if device.index is not None else 0
        torch.cuda.set_device(gpu_id)
        if is_main_process:
            print(f"[DEBUG] 模型构建前确认: 使用GPU {torch.cuda.current_device()}")
    
    # 构建模型
    model = build_model(config, src_vocab, tgt_vocab, device)
    
    # 验证模型参数所在设备
    if device.type == 'cuda' and is_main_process:
        first_param = next(model.parameters())
        actual_device = first_param.device
        print(f"[DEBUG] 模型参数实际所在设备: {actual_device}")
        if actual_device.index != device.index:
            print(f"[WARNING] 设备不匹配！期望 {device}，实际 {actual_device}")
        else:
            print(f"[DEBUG] ✅ 设备匹配正确，模型在 {actual_device} 上")
            # 显示GPU内存使用
            gpu_mem = torch.cuda.memory_allocated(gpu_id) / 1024**2
            print(f"[DEBUG] GPU{gpu_id}内存使用: {gpu_mem:.2f} MB")
    
    # 分布式训练：使用DDP包装模型
    if use_ddp and world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        if is_main_process:
            print("模型已使用DistributedDataParallel包装")
    
    # 定义损失函数（忽略padding）
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.PAD_IDX)
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # 创建训练器
    if is_main_process:
        print("\n" + "=" * 80)
        print("训练阶段")
        print("=" * 80)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config,
        checkpoint_dir=config['training']['save_dir']
    )
    
    # 设置词汇表（用于BLEU计算）
    trainer.set_vocabs(src_vocab, tgt_vocab)
    
    # 检查是否从checkpoint恢复
    resume_from_checkpoint = config['training'].get('resume_from_checkpoint', True)
    resume_ask_user = config['training'].get('resume_ask_user', False)
    start_epoch = 0
    
    # 如果设置为false，确保从头开始训练（清空损失历史，但不删除旧checkpoint）
    if not resume_from_checkpoint and is_main_process:
        trainer.train_losses = []
        trainer.valid_losses = []
        trainer.train_ppls = []
        trainer.valid_ppls = []
        trainer.valid_bleus = []
        trainer.global_batch_idx = 0
        trainer.current_epoch = 0
        trainer.best_valid_loss = float('inf')
        print("[INFO] resume_from_checkpoint=false, starting fresh training (old checkpoints preserved)")
    
    # 分布式训练：所有进程都需要知道start_epoch，但只有主进程加载checkpoint
    # 注意：只有当resume_from_checkpoint=true时才加载checkpoint
    if resume_from_checkpoint and is_main_process:
        checkpoint_path = os.path.join(config['training']['save_dir'], 'best_model.pt')
        if os.path.exists(checkpoint_path):
            compatible, checkpoint = trainer._check_checkpoint_compatibility(checkpoint_path)
            
            if compatible:
                if resume_ask_user:
                    # 交互式询问
                    response = input(f"\n发现兼容的checkpoint (epoch {checkpoint['epoch'] + 1})，是否继续训练？(y/n): ")
                    should_resume = response.lower() == 'y'
                else:
                    # 自动恢复
                    should_resume = True
                    print(f"\n[INFO] 发现兼容的checkpoint，自动从epoch {checkpoint['epoch'] + 1}继续训练")
                
                if should_resume:
                    trainer.load_checkpoint('best_model.pt', load_loss_history=True)
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"[INFO] 从epoch {start_epoch}继续训练")
            else:
                print(f"\n[WARNING] 发现checkpoint但配置不匹配，将从头开始训练")
                if resume_ask_user:
                    response = input("是否删除旧checkpoint？(y/n): ")
                    if response.lower() == 'y':
                        import shutil
                        shutil.rmtree(config['training']['save_dir'], ignore_errors=True)
                        os.makedirs(config['training']['save_dir'], exist_ok=True)
                        print("[INFO] 已删除旧checkpoint")
    
    # 分布式训练：同步start_epoch到所有进程
    if use_ddp and world_size > 1:
        start_epoch_tensor = torch.tensor([start_epoch], dtype=torch.long, device=device)
        dist.broadcast(start_epoch_tensor, src=0)
        start_epoch = start_epoch_tensor.item()
        if not is_main_process:
            # 非主进程也需要加载checkpoint（但不加载损失历史）
            checkpoint_path = os.path.join(config['training']['save_dir'], 'best_model.pt')
            if os.path.exists(checkpoint_path) and start_epoch > 0:
                try:
                    trainer.load_checkpoint('best_model.pt', load_loss_history=False)
                except Exception as e:
                    print(f"[WARNING] 进程{rank}加载checkpoint失败: {e}")
    
    # 开始训练
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        teacher_forcing_ratio=config['training']['teacher_forcing_ratio'],
        start_epoch=start_epoch
    )
    
    # 清理分布式环境
    if use_ddp and world_size > 1:
        cleanup_distributed()
    
    if is_main_process:
        print("\n训练完成！模型已保存。")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练神经机器翻译模型')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--rank', type=int, default=0, help='进程rank（分布式训练）')
    parser.add_argument('--world_size', type=int, default=None, help='总进程数（分布式训练）')
    parser.add_argument('--use_torchrun', action='store_true', help='使用torchrun启动（推荐）')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    use_ddp = config['training'].get('use_ddp', False) and torch.cuda.is_available()
    
    if use_ddp and args.use_torchrun:
        # 使用torchrun启动（推荐方式）
        # 命令行: torchrun --nproc_per_node=2 train.py --use_torchrun
        # torchrun会自动设置LOCAL_RANK和WORLD_SIZE环境变量，并初始化分布式环境
        rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', config.get('world_size', 2)))
        
        print(f"\n[DEBUG] =========================================")
        print(f"[DEBUG] torchrun模式启动")
        print(f"[DEBUG] LOCAL_RANK={rank}, WORLD_SIZE={world_size}")
        print(f"[DEBUG] MASTER_ADDR={os.environ.get('MASTER_ADDR', 'None')}")
        print(f"[DEBUG] MASTER_PORT={os.environ.get('MASTER_PORT', 'None')}")
        print(f"[DEBUG] dist.is_initialized()={dist.is_initialized()}")
        print(f"[DEBUG] =========================================\n")
        
        # 初始化分布式环境（torchrun设置了环境变量，但需要手动初始化）
        if not dist.is_initialized():
            print(f"[DEBUG] 使用torchrun环境变量初始化分布式环境...")
            # 使用torchrun提供的环境变量初始化
            master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
            master_port = os.environ.get('MASTER_PORT', '29500')
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port
            
            # 使用环境变量中的RANK（torchrun设置的是LOCAL_RANK，但init_process_group需要RANK）
            # 对于单节点多GPU，RANK = LOCAL_RANK
            rank_for_init = int(os.environ.get('RANK', rank))
            world_size_for_init = int(os.environ.get('WORLD_SIZE', world_size))
            
            print(f"[DEBUG] 初始化参数: rank={rank_for_init}, world_size={world_size_for_init}")
            print(f"[DEBUG] MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
            
            dist.init_process_group(
                backend=config.get('dist_backend', 'nccl'),
                init_method=f'tcp://{master_addr}:{master_port}',
                rank=rank_for_init,
                world_size=world_size_for_init
            )
            torch.cuda.set_device(rank)
            print(f"[DEBUG] 分布式环境初始化完成")
        
        main(rank, world_size, args.config)
    elif use_ddp and args.world_size:
        # 手动指定world_size
        main(args.rank, args.world_size, args.config)
    else:
        # 单GPU训练
        main(config_path=args.config)

