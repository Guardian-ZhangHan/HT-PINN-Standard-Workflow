"""
HT-PINN 结果可视化脚本
水文地质学顶刊标准绘图模块
对标 WRR、Journal of Hydrology 等期刊格式要求

作者: Guardian-ZhangHan
版本: 1.0.0
日期: 2026-05-16
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# ====================== 全局配置（顶刊标准） ======================
# 字体配置（所有顶刊要求的 Times New Roman）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# 分辨率配置（投稿标准 300dpi）
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 颜色映射配置（水文地学领域标准配色）
CMAP_K = 'jet'  # 渗透系数场标准配色
CMAP_H = 'viridis'  # 水头场标准配色

# 图件尺寸配置（2×2子图黄金比例）
FIGSIZE = (10, 8)

# 路径配置
MODEL_COEFF_DIR = '../model_coeff'
RESULTS_DIR = '../results'

# ====================== 工具函数 ======================
def setup_logging(log_file: str = '../logs/plot_results.log') -> None:
    """
    配置日志系统
    Args:
        log_file: 日志文件路径
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def create_output_dirs() -> None:
    """自动创建必要的输出文件夹"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logging.info(f"输出文件夹已确认: {RESULTS_DIR}")

# ====================== 数据加载函数 ======================
def load_results() -> Tuple[np.ndarray, ...]:
    """
    加载模型反演结果
    Returns:
        X: x坐标网格
        Y: y坐标网格
        K_true: 真实渗透系数场
        K_pred: 预测渗透系数场
        u_true: 真实水头场
        u_pred: 预测水头场
        x_obs: 观测点x坐标
        y_obs: 观测点y坐标
    """
    logging.info("开始加载模型结果文件...")
    
    try:
        K_true = np.loadtxt(os.path.join(MODEL_COEFF_DIR, 'K_true.txt'))
        u_true = np.loadtxt(os.path.join(MODEL_COEFF_DIR, 'u_true.txt'))
        X = np.loadtxt(os.path.join(MODEL_COEFF_DIR, 'X.txt'))
        Y = np.loadtxt(os.path.join(MODEL_COEFF_DIR, 'Y.txt'))
        K_pred = np.loadtxt(os.path.join(MODEL_COEFF_DIR, 'model_K.txt'))
        u_pred = np.loadtxt(os.path.join(MODEL_COEFF_DIR, 'model_u.txt'))
        x_obs = np.loadtxt(os.path.join(MODEL_COEFF_DIR, 'x_obs.txt'))
        y_obs = np.loadtxt(os.path.join(MODEL_COEFF_DIR, 'y_obs.txt'))
        
        logging.info("所有结果文件加载成功")
        return X, Y, K_true, K_pred, u_true, u_pred, x_obs, y_obs
    
    except FileNotFoundError as e:
        logging.error(f"结果文件缺失: {e}")
        logging.error("请先运行 run_forward.py 和 run_inverse.py 生成结果文件")
        raise

# ====================== 绘图函数 ======================
def plot_comparison(X: np.ndarray, Y: np.ndarray, 
                   K_true: np.ndarray, K_pred: np.ndarray,
                   u_true: np.ndarray, u_pred: np.ndarray,
                   x_obs: np.ndarray, y_obs: np.ndarray) -> plt.Figure:
    """
    绘制2×2对比图（顶刊标准布局）
    Args:
        X, Y: 坐标网格
        K_true, K_pred: 真实与预测渗透系数场
        u_true, u_pred: 真实与预测水头场
        x_obs, y_obs: 观测点坐标
    Returns:
        fig: matplotlib Figure对象
    """
    logging.info("开始绘制对比图...")
    
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    
    # 子图(a): 真实lnK场
    im1 = axes[0,0].pcolormesh(X, Y, K_true.reshape(X.shape), 
                              cmap=CMAP_K, vmin=-2, vmax=1.5)
    axes[0,0].set_title('(a) True ln$K$ field', fontweight='bold')
    axes[0,0].set_xlabel('x (m)')
    axes[0,0].set_ylabel('y (m)')
    fig.colorbar(im1, ax=axes[0,0], fraction=0.046, pad=0.04)
    
    # 子图(b): 反演lnK场（带观测点标记）
    im2 = axes[0,1].pcolormesh(X, Y, K_pred.reshape(X.shape), 
                              cmap=CMAP_K, vmin=-2, vmax=1.5)
    axes[0,1].scatter(x_obs, y_obs, c='black', s=20, marker='^', 
                     edgecolors='white', linewidths=0.5, label='Observation wells')
    axes[0,1].set_title('(b) Inverted ln$K$ field', fontweight='bold')
    axes[0,1].set_xlabel('x (m)')
    axes[0,1].set_ylabel('y (m)')
    axes[0,1].legend(fontsize=10)
    fig.colorbar(im2, ax=axes[0,1], fraction=0.046, pad=0.04)
    
    # 子图(c): 真实水头场
    im3 = axes[1,0].pcolormesh(X, Y, u_true.reshape(X.shape), cmap=CMAP_H)
    axes[1,0].scatter(0, 0, c='red', s=60, marker='s', 
                     edgecolors='white', linewidths=1, label='Source term')
    axes[1,0].set_title('(c) True hydraulic head', fontweight='bold')
    axes[1,0].set_xlabel('x (m)')
    axes[1,0].set_ylabel('y (m)')
    axes[1,0].legend(fontsize=10)
    fig.colorbar(im3, ax=axes[1,0], fraction=0.046, pad=0.04)
    
    # 子图(d): 预测水头场
    im4 = axes[1,1].pcolormesh(X, Y, u_pred.reshape(X.shape), cmap=CMAP_H)
    axes[1,1].set_title('(d) Predicted hydraulic head', fontweight='bold')
    axes[1,1].set_xlabel('x (m)')
    axes[1,1].set_ylabel('y (m)')
    fig.colorbar(im4, ax=axes[1,1], fraction=0.046, pad=0.04)
    
    # 自动调整布局，避免标签重叠
    plt.tight_layout()
    
    logging.info("对比图绘制完成")
    return fig

def save_figure(fig: plt.Figure, filename: str = 'HT_PINN_results') -> None:
    """
    保存图件为多种格式（顶刊要求）
    Args:
        fig: matplotlib Figure对象
        filename: 输出文件名（不含扩展名）
    """
    logging.info("开始保存图件...")
    
    # 保存PNG格式（用于README和PPT）
    png_path = os.path.join(RESULTS_DIR, f'{filename}.png')
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0.1)
    logging.info(f"PNG图件已保存: {png_path}")
    
    # 保存PDF格式（用于论文投稿，矢量图）
    pdf_path = os.path.join(RESULTS_DIR, f'{filename}.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.1)
    logging.info(f"PDF矢量图已保存: {pdf_path}")
    
    # 保存SVG格式（用于后期编辑）
    svg_path = os.path.join(RESULTS_DIR, f'{filename}.svg')
    fig.savefig(svg_path, bbox_inches='tight', pad_inches=0.1)
    logging.info(f"SVG矢量图已保存: {svg_path}")

# ====================== 主函数 ======================
def main() -> None:
    """主函数：执行完整的结果可视化流程"""
    try:
        # 初始化
        setup_logging()
        create_output_dirs()
        
        # 加载数据
        results = load_results()
        
        # 绘制对比图
        fig = plot_comparison(*results)
        
        # 保存结果
        save_figure(fig)
        
        # 显示图件（可选）
        # plt.show()
        
        logging.info("✅ 所有操作完成！顶刊级对比图已生成")
        logging.info(f"结果文件位于: {os.path.abspath(RESULTS_DIR)}")
        
    except Exception as e:
        logging.error(f"程序执行失败: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()