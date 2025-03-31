import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

def run_inpainting(checkpoint_path, output_dir):
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
        "--sweet-spot", "8",
        "--total-iter", "12",
        "--mask-func", "linear",
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
    checkpoint_dir = "/home/raiso/lab3/checkpoints/20250331_084501"
    epochs = []
    fid_scores = []
    
    # 遍歷所有 epoch checkpoints
    for file in tqdm(sorted(os.listdir(checkpoint_dir))):
        if file.startswith("epoch_") and file.endswith(".pth"):
            epoch = int(file.split("_")[1].split(".")[0])
            checkpoint_path = os.path.join(checkpoint_dir, file)
            output_dir = os.path.abspath(f"{checkpoint_dir}/inpainting_results_epoch_{epoch}")
            
            print(f"\n處理 epoch {epoch}...")
            
            # 執行 inpainting
            run_inpainting(checkpoint_path, output_dir)
            
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
    plt.title('FID Score vs. Epoch')
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
        writer.writerow(['Epoch', 'FID_Score'])  # 寫入標題行
        writer.writerows(data)  # 寫入數據
    
    print(f"FID scores 已保存為 CSV 文件: {csv_path}")

if __name__ == "__main__":
    main() 