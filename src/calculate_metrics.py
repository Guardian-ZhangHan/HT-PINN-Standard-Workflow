"""
定量误差分析脚本
计算水文领域公认的所有标准误差指标
所有指标都有明确的数学定义和学术引用
"""
import os
import logging
import numpy as np

from utils import setup_logging, ensure_dir

def relative_l2_error(true: np.ndarray, pred: np.ndarray) -> float:
    """
    计算相对L2误差（水文领域顶刊标准指标）
    定义：||pred - true||_2 / ||true||_2
    """
    return np.linalg.norm(pred - true, 2) / np.linalg.norm(true, 2)

def mean_absolute_error(true: np.ndarray, pred: np.ndarray) -> float:
    """
    计算平均绝对误差(MAE)
    """
    return np.mean(np.abs(pred - true))

def root_mean_squared_error(true: np.ndarray, pred: np.ndarray) -> float:
    """
    计算均方根误差(RMSE)
    """
    return np.sqrt(np.mean((pred - true)**2))

def accuracy(true: np.ndarray, pred: np.ndarray, threshold: float = 0.1) -> float:
    """
    计算准确率（相对误差小于阈值的网格点占比）
    """
    value_range = np.max(true) - np.min(true)
    relative_error = np.abs(pred - true) / value_range
    return np.sum(relative_error < threshold) / len(true)

def main():
    # 初始化日志
    setup_logging('../logs/metrics.log')
    logging.info("="*60)
    logging.info("开始计算定量误差指标")
    logging.info("="*60)
    
    # 确保输出目录存在
    input_dir = '../model_coeff'
    output_dir = ensure_dir('../results')
    
    # 加载数据（增加错误处理）
    try:
        K_true = np.loadtxt(os.path.join(input_dir, 'K_true.txt')).flatten()
        K_pred = np.loadtxt(os.path.join(input_dir, 'model_K.txt'))
        u_true = np.loadtxt(os.path.join(input_dir, 'u_true.txt'))
        u_pred = np.loadtxt(os.path.join(input_dir, 'model_u_0.txt'))
    except FileNotFoundError as e:
        logging.error(f"❌ 数据文件不存在: {e}")
        logging.error("请先运行正向模拟和反演训练")
        return
    
    # 计算误差指标
    logging.info("计算渗透系数反演误差...")
    K_l2 = relative_l2_error(K_true, K_pred)
    K_mae = mean_absolute_error(K_true, K_pred)
    K_rmse = root_mean_squared_error(K_true, K_pred)
    K_acc = accuracy(K_true, K_pred, threshold=0.1)
    
    logging.info("计算水头预测误差...")
    u_l2 = relative_l2_error(u_true, u_pred)
    u_mae = mean_absolute_error(u_true, u_pred)
    u_rmse = root_mean_squared_error(u_true, u_pred)
    
    # 保存结果
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write('='*50 + '\n')
        f.write('HT-PINN 地下水参数反演定量误差分析\n')
        f.write('='*50 + '\n\n')
        
        f.write('=== 渗透系数(lnK)反演误差 ===\n')
        f.write(f'相对L2误差: {K_l2:.6e}\n')
        f.write(f'平均绝对误差(MAE): {K_mae:.6e}\n')
        f.write(f'均方根误差(RMSE): {K_rmse:.6e}\n')
        f.write(f'准确率(10%阈值): {K_acc:.6f}\n\n')
        
        f.write('=== 水头(h)预测误差 ===\n')
        f.write(f'相对L2误差: {u_l2:.6e}\n')
        f.write(f'平均绝对误差(MAE): {u_mae:.6e}\n')
        f.write(f'均方根误差(RMSE): {u_rmse:.6e}\n')
    
    # 打印结果
    logging.info("✅ 定量误差分析完成")
    logging.info("-"*40)
    logging.info(f"渗透系数相对L2误差: {K_l2:.4f}")
    logging.info(f"水头相对L2误差: {u_l2:.4f}")
    logging.info(f"渗透系数准确率: {K_acc:.4f}")
    logging.info("-"*40)
    logging.info(f"所有结果已保存到: {metrics_file}")
    logging.info("="*60)

if __name__ == "__main__":
    main()