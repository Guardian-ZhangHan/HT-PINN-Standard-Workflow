"""
顶刊级结果可视化脚本
生成符合Nature、WRR、JH等顶刊规范的对比图
使用感知均匀颜色映射，避免色盲问题
"""
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import warnings

from utils import setup_logging, ensure_dir

# 忽略matplotlib警告
warnings.filterwarnings('ignore')

def set_academic_style():
    """
    设置学术绘图全局样式
    严格参考Nature和WRR期刊的绘图规范
    """
    plt.rcParams.update({
        # 字体设置（顶刊标准）
        'font.family': 'Times New Roman',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        
        # 线条和边框设置
        'axes.linewidth': 1.0,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        
        # 分辨率设置
        'figure.dpi': 300,
        'savefig.dpi': 300,
        
        # 布局设置
        'figure.autolayout': True,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # 颜色设置（使用Nature推荐的感知均匀颜色映射）
        'axes.prop_cycle': plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    })

def main():
    # 初始化日志
    setup_logging('../logs/plot.log')
    logging.info("="*60)
    logging.info("开始生成顶刊级对比图")
    logging.info("="*60)
    
    # 设置学术绘图样式
    set_academic_style()
    
    # 确保输出目录存在
    input_dir = '../model_coeff'
    output_dir = ensure_dir('../results')
    
    # 加载数据
    try:
        K_true = np.loadtxt(os.path.join(input_dir, 'K_true.txt'))
        u_true = np.loadtxt(os.path.join(input_dir, 'u_true.txt'))
        X = np.loadtxt(os.path.join(input_dir, 'X.txt'))
        Y = np.loadtxt(os.path.join(input_dir, 'Y.txt'))
        K_pred = np.loadtxt(os.path.join(input_dir, 'model_K.txt')).reshape(X.shape)
        u_pred = np.loadtxt(os.path.join(input_dir, 'model_u_0.txt')).reshape(X.shape)
        x_obs = np.loadtxt(os.path.join(input_dir, 'x_obs.txt'))
        y_obs = np.loadtxt(os.path.join(input_dir, 'y_obs.txt'))
    except FileNotFoundError as e:
        logging.error(f"❌ 数据文件不存在: {e}")
        logging.error("请先运行正向模拟和反演训练")
        return
    
    # 创建2×2子图
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # 统一颜色范围（真实场和反演场使用相同刻度，方便对比）
    vmin_K = np.min(K_true)
    vmax_K = np.max(K_true)
    vmin_u = np.min(u_true)
    vmax_u = np.max(u_true)
    
    # (a) 真实渗透系数场
    im1 = axes[0, 0].pcolormesh(X, Y, K_true, cmap='viridis', vmin=vmin_K, vmax=vmax_K, shading='auto')
    axes[0, 0].set_title('(a) True ln$K$ field', fontweight='bold', pad=10)
    axes[0, 0].set_xlabel('x (m)')
    axes[0, 0].set_ylabel('y (m)')
    axes[0, 0].set_aspect('equal')
    cbar1 = fig.colorbar(im1, ax=axes[0, 0], fraction=0.045, pad=0.04)
    cbar1.set_label('ln$K$ (m/d)', rotation=270, labelpad=15)
    
    # (b) 反演渗透系数场
    im2 = axes[0, 1].pcolormesh(X, Y, K_pred, cmap='viridis', vmin=vmin_K, vmax=vmax_K, shading='auto')
    axes[0, 1].scatter(x_obs, y_obs, c='black', s=15, marker='^', 
                       edgecolors='white', linewidths=0.5, label='Observation points')
    axes[0, 1].set_title('(b) Inverted ln$K$ field', fontweight='bold', pad=10)
    axes[0, 1].set_xlabel('x (m)')
    axes[0, 1].set_ylabel('y (m)')
    axes[0, 1].set_aspect('equal')
    cbar2 = fig.colorbar(im2, ax=axes[0, 1], fraction=0.045, pad=0.04)
    cbar2.set_label('ln$K$ (m/d)', rotation=270, labelpad=15)
    axes[0, 1].legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # (c) 真实水头场
    im3 = axes[1, 0].pcolormesh(X, Y, u_true, cmap='plasma', vmin=vmin_u, vmax=vmax_u, shading='auto')
    axes[1, 0].scatter(0, 0, c='red', s=50, marker='s', 
                       edgecolors='white', linewidths=1, label='Pumping well')
    axes[1, 0].set_title('(c) True hydraulic head', fontweight='bold', pad=10)
    axes[1, 0].set_xlabel('x (m)')
    axes[1, 0].set_ylabel('y (m)')
    axes[1, 0].set_aspect('equal')
    cbar3 = fig.colorbar(im3, ax=axes[1, 0], fraction=0.045, pad=0.04)
    cbar3.set_label('Head (m)', rotation=270, labelpad=15)
    axes[1, 0].legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # (d) 预测水头场
    im4 = axes[1, 1].pcolormesh(X, Y, u_pred, cmap='plasma', vmin=vmin_u, vmax=vmax_u, shading='auto')
    axes[1, 1].set_title('(d) Predicted hydraulic head', fontweight='bold', pad=10)
    axes[1, 1].set_xlabel('x (m)')
    axes[1, 1].set_ylabel('y (m)')
    axes[1, 1].set_aspect('equal')
    cbar4 = fig.colorbar(im4, ax=axes[1, 1], fraction=0.045, pad=0.04)
    cbar4.set_label('Head (m)', rotation=270, labelpad=15)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片（顶刊要求同时提供矢量图和高分辨率位图）
    pdf_path = os.path.join(output_dir, 'HT_PINN_results.pdf')
    png_path = os.path.join(output_dir, 'HT_PINN_results.png')
    
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', transparent=False)
    
    logging.info(f"✅ 顶刊级对比图生成成功")
    logging.info(f"矢量图已保存到: {pdf_path}")
    logging.info(f"位图已保存到: {png_path}")
    logging.info("="*60)

if __name__ == "__main__":
    main()