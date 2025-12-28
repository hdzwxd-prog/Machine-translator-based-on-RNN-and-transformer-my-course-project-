#!/bin/bash
# 分布式训练启动脚本（双A6000 GPU）
# 
# 调试说明：
# - 如果卡住，检查端口是否被占用：netstat -tuln | grep 29501
# - 如果端口冲突，修改--master_port参数
# - 确保没有其他训练进程在运行：ps aux | grep train.py

# 设置OMP线程数（避免警告）
export OMP_NUM_THREADS=4

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1

# 检查并清理旧的训练进程
echo "[DEBUG] 检查是否有旧的训练进程..."
OLD_PROCESSES=$(ps aux | grep -E "train.py --use_torchrun|torchrun.*train.py" | grep -v grep | wc -l)
if [ "$OLD_PROCESSES" -gt 0 ]; then
    echo "[WARNING] 发现 $OLD_PROCESSES 个旧的训练进程，正在清理..."
    pkill -f "train.py --use_torchrun" 2>/dev/null
    pkill -f "torchrun.*train.py" 2>/dev/null
    sleep 2
    echo "[INFO] 清理完成"
fi

# 检查GPU是否可用
echo "[DEBUG] 检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || echo "[WARNING] nvidia-smi不可用"

# 自动检测可用端口
MASTER_PORT=29501
while netstat -tuln 2>/dev/null | grep -q ":$MASTER_PORT " || ss -tuln 2>/dev/null | grep -q ":$MASTER_PORT "; do
    echo "[WARNING] 端口 $MASTER_PORT 已被占用，尝试下一个端口..."
    MASTER_PORT=$((MASTER_PORT + 1))
    if [ $MASTER_PORT -gt 30000 ]; then
        echo "[ERROR] 无法找到可用端口"
        exit 1
    fi
done

# 使用torchrun启动分布式训练（推荐方式）
# --nproc_per_node=2: 使用2个GPU
# --master_port: 主进程端口（自动检测）
echo "[DEBUG] 启动分布式训练..."
echo "[DEBUG] 使用端口: $MASTER_PORT"
torchrun --nproc_per_node=2 --master_port=$MASTER_PORT train.py --use_torchrun

# 或者使用python直接启动（备选方式）
# python train.py --rank 0 --world_size 2 &
# python train.py --rank 1 --world_size 2 &
# wait

echo "训练完成！"

