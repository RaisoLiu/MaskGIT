#!/bin/bash
source .venv/bin/activate

# 檢查是否提供了輸入路徑
if [ -z "$1" ]; then
    echo "請提供 inpaint_result 資料夾路徑"
    echo "使用方式: ./calculate_fid.sh /path/to/inpaint_result"
    exit 1
fi

INPAINT_PATH=$1

# 檢查資料夾是否存在
if [ ! -d "$INPAINT_PATH" ]; then
    echo "錯誤：資料夾 $INPAINT_PATH 不存在"
    exit 1
fi

# 設定輸出檔案路徑
OUTPUT_FILE="$INPAINT_PATH/fid_score.txt"

# 計算 FID score 並只保存最終結果
echo "計算 FID score..."
cd faster-pytorch-fid/
python fid_score_gpu.py --predicted-path "$INPAINT_PATH/test_results" --device cuda:0 2>&1 | grep "FID:" > "$OUTPUT_FILE"


echo "FID score 已保存到 $OUTPUT_FILE" 

cat "$OUTPUT_FILE"