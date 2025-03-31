#!/bin/bash

# 啟動虛擬環境
source .venv/bin/activate

# 設置 CUDA 可見設備（如果需要）
# export CUDA_VISIBLE_DEVICES=0

# 檢查必要的目錄是否存在
if [ ! -d "./lab3_dataset" ]; then
    echo "錯誤：找不到 lab3_dataset 目錄"
    exit 1
fi

if [ ! -d "./config" ]; then
    echo "錯誤：找不到 config 目錄"
    exit 1
fi

# 訓練參數設置
TRAIN_DATA_PATH="./lab3_dataset/train"
VAL_DATA_PATH="./lab3_dataset/val"
OUT_DIR_PATH="./checkpoints"
DEVICE="cuda:0"
NUM_WORKERS=4
BATCH_SIZE=48
PARTIAL=1.0
ACCUM_GRAD=1
EPOCHS=10
SAVE_PER_EPOCH=3
START_FROM_EPOCH=0
LEARNING_RATE=1e-4
CONFIG_PATH="config/MaskGit.yml"
WARMUP_EPOCHS=10

# 創建必要的目錄
mkdir -p $OUT_DIR_PATH

# 檢查配置文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
    echo "錯誤：找不到配置文件 $CONFIG_PATH"
    exit 1
fi

# 執行訓練腳本
python training_transformer.py \
    --train_d_path $TRAIN_DATA_PATH \
    --val_d_path $VAL_DATA_PATH \
    --out_d_path $OUT_DIR_PATH \
    --device $DEVICE \
    --num_workers $NUM_WORKERS \
    --batch-size $BATCH_SIZE \
    --partial $PARTIAL \
    --accum-grad $ACCUM_GRAD \
    --epochs $EPOCHS \
    --save-per-epoch $SAVE_PER_EPOCH \
    --start-from-epoch $START_FROM_EPOCH \
    --learning-rate $LEARNING_RATE \
    --MaskGitConfig $CONFIG_PATH \
    --warmup-epochs $WARMUP_EPOCHS

# 退出虛擬環境
deactivate