#!/bin/bash

# 确保输出及时刷新
export PYTHONUNBUFFERED=1

# 设置线程数，控制 OpenMP 并行线程
export OMP_NUM_THREADS=1

# 设置默认通信端口（如未传递环境变量 PORT，则默认为 29500）
PORT=${PORT:-29500}

# 打印端口号，便于调试
echo "Using master port: $PORT"

# 设置 PYTHONPATH，确保脚本依赖的模块可以被找到
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# 启动分布式训练
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
    $(dirname "$0")/train.py --launcher pytorch "${@:3}"