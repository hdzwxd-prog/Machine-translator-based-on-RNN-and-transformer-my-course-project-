#!/bin/bash
# 单GPU训练启动脚本

# 使用第一个GPU
export CUDA_VISIBLE_DEVICES=0

# 单GPU训练
python train.py

echo "训练完成！"

