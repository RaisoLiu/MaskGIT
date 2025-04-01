import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob

def read_fid_scores(folder_path):
    # 讀取資料夾中的 FID score CSV 文件
    csv_file = os.path.join(folder_path, 'fid_scores.csv')
    if not os.path.exists(csv_file):
        print(f"警告：找不到文件 {csv_file}")
        return None, None
        
    # 讀取 CSV 文件
    df = pd.read_csv(csv_file)
    
    # 假設 CSV 文件有 'epoch' 和 'fid_score' 列
    epochs = df['Epoch'].values
    scores = df['FID_Score'].values
    
    return epochs, scores

def plot_fid_comparison(folders):
    # 確保 output 資料夾存在
    os.makedirs('output', exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # 定義顏色和標記
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']
    markers = ['o', 's', '^', 'D', 'v']
    mask_type_list = ["cosine","linear","linear-half","sine_linear","sqrt","square"]
    
    for i, folder in enumerate(folders):
        if not os.path.exists(folder):
            print(f"警告：資料夾 {folder} 不存在")
            continue
            
        epochs, scores = read_fid_scores(folder)
        if epochs is None or scores is None:
            continue
            
        mask_type = mask_type_list[i]
        
        plt.plot(epochs, scores, 
                color=colors[i % len(colors)], 
                marker=markers[i % len(markers)],
                label=mask_type,
                linewidth=2,
                markersize=6)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('FID Score', fontsize=12)
    plt.title('FID Score Comparison Across Different Mask Functions', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # 保存 PNG 格式到 output 資料夾
    plt.savefig(os.path.join('output', 'fid_comparison.png'), dpi=300, bbox_inches='tight')
    
    # 保存 EPS 格式到 output 資料夾
    plt.savefig(os.path.join('output', 'fid_comparison.eps'), format='eps', bbox_inches='tight')
    
    plt.close()

if __name__ == "__main__":
    # 指定要比較的資料夾
    folders = [
        "checkpoints/20250331_113914-cosine",
        "checkpoints/20250331_113914-linear",
        "checkpoints/20250331_113914-linear-10-20",
        "checkpoints/20250331_113914-sine_linear",
        "checkpoints/20250331_113914-sqrt",
        "checkpoints/20250331_113914-square"
    ]
    
    plot_fid_comparison(folders) 