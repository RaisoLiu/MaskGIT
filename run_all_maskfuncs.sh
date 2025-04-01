#!/bin/bash

# # 檢查是否提供了 checkpoint 目錄參數
# if [ $# -ne 1 ]; then
#     echo "使用方式: $0 <checkpoint_directory>"
#     echo "例如: $0 /home/raiso/lab3/checkpoints/20250331_113914-cosine"
#     exit 1
# fi

# CHECKPOINT_DIR=$1

# 定義所有要測試的 mask functions
# MASK_FUNCS=("cosine" "linear" "square" "sqrt" "constant")
FOLDER_NAMES=("/home/raiso/lab3/checkpoints/checkpoints/20250331_201510-sine_linear-8-12")
MASK_FUNCS=("sine_linear")

# 設定其他參數
TOTAL_ITER=12
SWEET_SPOT=8

# 遍歷所有 mask functions
for i in "${!FOLDER_NAMES[@]}"; do
    folder_name=${FOLDER_NAMES[$i]}
    mask_func=${MASK_FUNCS[$i]}
    
    echo "執行 mask function: $folder_name"
    python plot_fid_epochs.py "$folder_name" \
        --mask-func "$mask_func" \
        --total-iter "$TOTAL_ITER" \
        --sweet-spot "$SWEET_SPOT"
    
    echo "完成 $mask_func 的計算"
    echo "----------------------------------------"
done

echo "所有 mask functions 的計算已完成！" 