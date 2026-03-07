#!/bin/bash

# --- 1. 环境配置 ---
# 解决显存碎片化问题，优化显存分配
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 设置分布式训练地址 (通常 torchrun 会自动处理，但手动指定更稳健)
export MASTER_ADDR=localhost
export MASTER_PORT=12345

# --- 2. 训练参数设置 ---
GPUS_PER_NODE=4          # 总卡数
BATCH_SIZE=2             # 建议设为 8，这样总 batch 就是 8*4=32
CROP_SIZE=448            # 训练裁剪尺寸
CROP_NUMS=10              

MAX_ITERS=10000
LOG_ITERS=500
EVAL_ITERS=500
WARMUP_ITERS=1500
SAVE_ITERS=2000

SEED=33
LR=6e-5
TEMP=0.5

# LOCAL_WEIGHTS=/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/runs/cam/2026-0304-1412-12/checkpoints/model_iter_4000.pth
LOCAL_WEIGHTS=/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/runs/cam/2026-0304-0020-09/checkpoints/model_iter_4000.pth


# --- 3. 启动命令 ---
# 使用 torchrun 启动，它会自动设置 RANK 和 WORLD_SIZE
torchrun --nproc_per_node=$GPUS_PER_NODE \
    CAM_SAM/train.py \
    --batch_size $BATCH_SIZE \
    --crop_size $CROP_SIZE \
    --lr $LR \
    --max_iters $MAX_ITERS \
    --log_iters $LOG_ITERS \
    --eval_iters $EVAL_ITERS \
    --warmup_iters $WARMUP_ITERS \
    --save_iters $SAVE_ITERS \
    --seed $SEED \
    --temp $TEMP \
    --backbone vit_base_patch16_224 \
    --bkg_thre 0.45 \
    --high_thre 0.75 \
    --low_thre 0.25 \
    --weights $LOCAL_WEIGHTS \