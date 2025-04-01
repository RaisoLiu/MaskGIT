#!/bin/bash

source .venv/bin/activate

# 設定 CUDA 裝置
export CUDA_VISIBLE_DEVICES=0

# 執行 inpainting.py
python inpainting.py \
    --device cuda \
    --batch-size 1 \
    --partial 1.0 \
    --num_workers 4 \
    --MaskGitConfig config/MaskGit.yml \
    --load-transformer-ckpt-path /home/raiso/lab3/checkpoints/v100-epoch200/epoch_220.pth \
    --test-maskedimage-path ./lab3_dataset/masked_image \
    --test-mask-path ./lab3_dataset/mask64 \
    --sweet-spot 8 \
    --total-iter 12 \
    --mask-func cosine \
    --predicted-parent_path ./inpainting_results