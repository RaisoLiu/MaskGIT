import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from natsort import natsorted
import argparse

def run_inpainting(checkpoint_path, output_dir, mask_func, total_iter, sweet_spot):
    """執行 inpainting 腳本"""
    cmd = [
        "python", "inpainting.py",
        "--device", "cuda",
        "--batch-size", "1",
        "--partial", "1.0",
        "--num_workers", "4",
        "--MaskGitConfig", "config/MaskGit.yml",
        "--load-transformer-ckpt-path", checkpoint_path,
        "--test-maskedimage-path", "./lab3_dataset/masked_image",
        "--test-mask-path", "./lab3_dataset/mask64",
        "--sweet-spot", str(sweet_spot),
        "--total-iter", str(total_iter),
        "--mask-func", mask_func,
        "--predicted-path", output_dir
    ]
    subprocess.run(cmd, check=True)

def calculate_fid(output_dir):
    """計算 FID score"""
    # 確保使用絕對路徑
    output_dir_abs = os.path.abspath(os.path.join(output_dir, "test_results"))
    cmd = [
        "cd", "faster-pytorch-fid/",
        "&&",
        "python", "fid_score_gpu.py",
        "--predicted-path", output_dir_abs,
        "--device", "cuda:0"
    ]
    result = subprocess.run(" ".join(cmd), shell=True, capture_output=True, text=True)
    # 從輸出中提取 FID score
    for line in result.stdout.split('\n'):
        if "FID:" in line:
            return float(line.split("FID:")[1].strip())
    return None

def main():
    # 設定命令行參數
    parser = argparse.ArgumentParser(description='計算不同 epoch 的 FID score 並繪製圖表')
    parser.add_argument('checkpoint_dir', type=str, help='checkpoint 資料夾的路徑')
    parser.add_argument('--mask-func', type=str, default='linear',
                      choices=['linear', 'cosine', 'square', 'sqrt', 'sine_linear', 'constant'],
                      help='遮罩函數類型 (預設: linear)')
    parser.add_argument('--total-iter', type=int, default=12,
                      help='總迭代次數 (預設: 12)')
    parser.add_argument('--sweet-spot', type=int, default=12,
                      help='sweet spot 值 (預設: 12)')
    args = parser.parse_args()
    
    # 確保 checkpoint_dir 是絕對路徑
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    
    # 檢查資料夾是否存在
    if not os.path.exists(checkpoint_dir):
        print(f"錯誤：資料夾 {checkpoint_dir} 不存在")
        return
    
    print(f"使用參數：")
    print(f"- 檢查點路徑：{checkpoint_dir}")
    print(f"- 遮罩函數：{args.mask_func}")
    print(f"- 總迭代次數：{args.total_iter}")
    print(f"- Sweet spot：{args.sweet_spot}")
    
    epochs = []
    fid_scores = []
    
    # 遍歷所有 epoch checkpoints
    for file in tqdm(natsorted(os.listdir(checkpoint_dir))):
        if file.startswith("epoch_") and file.endswith(".pth"):
            epoch = int(file.split("_")[1].split(".")[0])
            checkpoint_path = os.path.join(checkpoint_dir, file)
            output_dir = os.path.abspath(f"{checkpoint_dir}/inpainting_results_epoch_{epoch}")
            
            print(f"\n處理 epoch {epoch}...")
            
            # 執行 inpainting
            run_inpainting(checkpoint_path, output_dir, args.mask_func, args.total_iter, args.sweet_spot)
            
            # 計算 FID score
            fid_score = calculate_fid(output_dir)
            
            if fid_score is not None:
                epochs.append(epoch)
                fid_scores.append(fid_score)
                print(f"Epoch {epoch}: FID = {fid_score}")
            else:
                print(f"警告：無法計算 epoch {epoch} 的 FID score")
    
    if not epochs:
        print("錯誤：沒有成功計算任何 FID score")
        return
    
    # 繪製圖表
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, fid_scores, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Epoch')
    plt.ylabel('FID Score')
    plt.title(f'FID Score vs. Epoch (mask_func={args.mask_func}, total_iter={args.total_iter}, sweet_spot={args.sweet_spot})')
    plt.grid(True)
    
    # 保存圖表
    plt.savefig(f'{checkpoint_dir}/fid_vs_epoch.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{checkpoint_dir}/fid_vs_epoch.eps', format='eps', bbox_inches='tight')
    plt.close()
    
    print(f"\n圖表已保存為 {checkpoint_dir}/fid_vs_epoch.png 和 {checkpoint_dir}/fid_vs_epoch.eps")

    # 將 FID score 存成 CSV
    import csv
    
    # 創建包含 epoch 和 fid_score 的數據
    data = list(zip(epochs, fid_scores))
    
    # 將數據寫入 CSV 文件
    csv_path = f'{checkpoint_dir}/fid_scores.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'FID_Score', 'Mask_Func', 'Total_Iter', 'Sweet_Spot'])  # 寫入標題行
        for row in data:
            writer.writerow([row[0], row[1], args.mask_func, args.total_iter, args.sweet_spot])  # 寫入數據
    
    print(f"FID scores 已保存為 CSV 文件: {csv_path}")

if __name__ == "__main__":
    main() 