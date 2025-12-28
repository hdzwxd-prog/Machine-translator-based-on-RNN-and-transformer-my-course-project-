#!/bin/bash
# 对比实验脚本
# 自动运行多组实验，对比不同配置的效果

echo "=========================================="
echo "神经机器翻译对比实验"
echo "=========================================="

# 创建实验结果目录
mkdir -p experiments

# 实验1：对比注意力机制
echo ""
echo "实验1：对比不同注意力机制"
echo "=========================================="

for attention in dot general additive; do
    echo ""
    echo "训练模型：注意力机制=$attention"
    
    # 修改配置文件
    sed -i "s/type: .*/type: \"$attention\"/" config/config.yaml
    
    # 训练模型
    python train.py
    
    # 重命名检查点
    mv checkpoints/best_model.pt experiments/best_model_attn_${attention}.pt
    
    # 评估模型（贪心解码）
    echo "评估模型（贪心解码）"
    python evaluate.py --checkpoint experiments/best_model_attn_${attention}.pt \
                       --strategy greedy \
                       --output experiments/results_attn_${attention}_greedy.txt
    
    # 评估模型（束搜索）
    echo "评估模型（束搜索）"
    python evaluate.py --checkpoint experiments/best_model_attn_${attention}.pt \
                       --strategy beam_search \
                       --beam_size 5 \
                       --output experiments/results_attn_${attention}_beam5.txt
done

# 实验2：对比Teacher Forcing比例
echo ""
echo "实验2：对比不同Teacher Forcing比例"
echo "=========================================="

# 恢复默认注意力机制
sed -i "s/type: .*/type: \"dot\"/" config/config.yaml

for tf_ratio in 1.0 0.5 0.0; do
    echo ""
    echo "训练模型：Teacher Forcing比例=$tf_ratio"
    
    # 修改配置文件
    sed -i "s/teacher_forcing_ratio: .*/teacher_forcing_ratio: $tf_ratio/" config/config.yaml
    
    # 训练模型
    python train.py
    
    # 重命名检查点
    mv checkpoints/best_model.pt experiments/best_model_tf_${tf_ratio}.pt
    
    # 评估模型
    python evaluate.py --checkpoint experiments/best_model_tf_${tf_ratio}.pt \
                       --strategy beam_search \
                       --beam_size 5 \
                       --output experiments/results_tf_${tf_ratio}.txt
done

# 实验3：对比解码策略（使用同一个模型）
echo ""
echo "实验3：对比不同解码策略"
echo "=========================================="

MODEL="experiments/best_model_attn_dot.pt"

echo "贪心解码"
python evaluate.py --checkpoint $MODEL \
                   --strategy greedy \
                   --output experiments/results_decode_greedy.txt

for beam_size in 3 5 10; do
    echo "束搜索解码：beam_size=$beam_size"
    python evaluate.py --checkpoint $MODEL \
                       --strategy beam_search \
                       --beam_size $beam_size \
                       --output experiments/results_decode_beam${beam_size}.txt
done

echo ""
echo "=========================================="
echo "所有实验完成！结果保存在 experiments/ 目录"
echo "=========================================="

