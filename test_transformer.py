"""
测试Transformer模型训练
注意：当前GPU正在训练RNN，此脚本使用较小的batch_size和单GPU以避免内存冲突
"""
import os
import sys
import torch

# 检查GPU内存
if torch.cuda.is_available():
    print(f"可用GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  总内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  已分配: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"  缓存: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    print()

# 设置使用GPU 1以避免与RNN训练冲突（GPU 0正在训练RNN）
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 使用第二个GPU

# 导入训练脚本
from train import main, load_config

def test_transformer_training():
    """测试Transformer模型训练"""
    print("=" * 80)
    print("测试Transformer模型训练")
    print("=" * 80)
    
    # 使用Transformer配置文件
    config_path = 'config/config_transformer.yaml'
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return False
    
    # 加载配置
    config = load_config(config_path)
    
    # 进一步减小batch_size以确保内存充足
    original_batch_size = config['training']['batch_size']
    config['training']['batch_size'] = min(original_batch_size, 16)
    print(f"使用batch_size: {config['training']['batch_size']} (原始: {original_batch_size})")
    
    # 禁用分布式训练
    config['training']['use_ddp'] = False
    config['world_size'] = 1
    
    # 只训练少量epoch进行测试
    original_epochs = config['training']['num_epochs']
    config['training']['num_epochs'] = 2  # 只训练2个epoch进行测试
    print(f"测试模式: 只训练 {config['training']['num_epochs']} 个epoch (原始: {original_epochs})")
    
    try:
        # 运行训练
        main(config_path=config_path)
        print("\n" + "=" * 80)
        print("Transformer模型训练测试成功！")
        print("=" * 80)
        return True
    except Exception as e:
        print(f"\n错误: Transformer模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_transformer_training()
    sys.exit(0 if success else 1)

