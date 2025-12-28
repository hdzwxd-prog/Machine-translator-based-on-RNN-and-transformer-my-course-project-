#!/bin/bash
# 分布式训练调试脚本

echo "=========================================="
echo "分布式训练调试信息"
echo "=========================================="

echo ""
echo "1. 检查GPU状态:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "  [ERROR] nvidia-smi不可用"

echo ""
echo "2. 检查端口占用:"
netstat -tuln | grep -E "29501|12355" || echo "  [OK] 端口未被占用"

echo ""
echo "3. 检查残留进程:"
ps aux | grep -E "train.py|torchrun" | grep -v grep || echo "  [OK] 没有残留进程"

echo ""
echo "4. 检查环境变量:"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-未设置}"
echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS:-未设置}"

echo ""
echo "5. 测试单GPU训练（快速验证）:"
echo "  运行: python train.py"
echo "  如果单GPU正常，问题可能在分布式初始化"

echo ""
echo "=========================================="
