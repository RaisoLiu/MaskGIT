import numpy as np
import matplotlib.pyplot as plt
import os

def get_gamma_func(mode="linear"):
    gamma_funcs = {
        "linear": lambda r: 1 - r,
        "cosine": lambda r: np.cos(r * np.pi / 2),
        "square": lambda r: 1 - r ** 2,
        "sqrt": lambda r: 1 - np.sqrt(r),
        "sine_linear": lambda r: (1 - r) * (1 - np.sin(2 * np.pi * r)  *  0.75),
        "constant": lambda r: np.zeros_like(r)
    }
    if mode not in gamma_funcs:
        raise NotImplementedError
    return gamma_funcs[mode]

def plot_gamma_functions():
    # 創建輸出目錄
    os.makedirs('output', exist_ok=True)
    
    # 生成 x 軸數據
    r = np.linspace(0, 1, 1000)
    
    # 獲取所有 gamma 函數類型
    gamma_types = ["linear", "cosine", "square", "sqrt", "sine_linear", "constant"]
    
    # 創建圖形
    plt.figure(figsize=(12, 8))
    
    # 繪製每個函數
    for gamma_type in gamma_types:
        gamma_func = get_gamma_func(gamma_type)
        y = gamma_func(r)
        plt.plot(r, y, label=gamma_type, linewidth=2)
    
    # 設置圖形屬性
    plt.title('Different Gamma Functions from MaskGit', fontsize=14)
    plt.xlabel('r', fontsize=12)
    plt.ylabel('γ(r)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # 保存圖形
    plt.savefig('output/gamma_functions.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig('output/gamma_functions.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_gamma_functions() 