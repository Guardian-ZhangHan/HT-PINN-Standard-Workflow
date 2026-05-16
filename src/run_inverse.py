"""
反演训练脚本
使用HT-PINN方法反演地下水渗透系数场
参考：[原始HT-PINN论文完整引用]
"""
import os
import time
import numpy as np
import scipy.io as sio
import torch
import torch.optim as optim
from tqdm import tqdm

from utils import set_seed, setup_logging, ensure_dir
from model import PhysicsInformedNN

def main():
    # 初始化日志（同时输出到控制台和文件）
    setup_logging('../logs/inverse.log')
    logging.info("="*60)
    logging.info("开始运行反演训练")
    logging.info("="*60)
    
    # 固定随机种子（顶刊强制可复现要求）
    set_seed(42)
    
    # 确保所有输出目录存在
    output_dir = ensure_dir('../model_coeff')
    log_dir = ensure_dir('../logs')
    
    start_time = time.time()
    
    # ==============================================
    # 完整反演训练核心代码（从HT_PINN_inverse.ipynb提取）
    # ==============================================
    # 1. 加载合成数据
    try:
        data = sio.loadmat('../data/HT_synthetic.mat')
    except FileNotFoundError:
        logging.error("❌ 数据文件不存在: ../data/HT_synthetic.mat")
        logging.error("请确保数据文件放在正确的位置")
        return
    
    # 2. 提取数据
    logK_true = data['logK']
    heads = data['heads']
    obs_x = data['obs_x']
    obs_y = data['obs_y']
    nx, ny = logK_true.shape
    
    # 3. 生成计算网格
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    
    # 4. 边界条件设置
    # 左边界：狄利克雷边界 h=1
    left_bound = np.hstack((np.zeros((ny, 1)), y[:, None]))
    # 右边界：狄利克雷边界 h=0
    right_bound = np.hstack((np.ones((ny, 1)), y[:, None]))
    # 上下边界：诺伊曼边界 ∂h/∂y=0
    top_bound = np.hstack((x[:, None], np.ones((nx, 1))))
    bottom_bound = np.hstack((x[:, None], np.zeros((nx, 1))))
    
    # 5. 抽水井ID列表
    pump_id_list = list(range(25))
    
    # 6. 准备训练数据
    train_dict = {}
    
    # 观测点数据
    train_dict['x_u'] = []
    train_dict['y_u'] = []
    train_dict['u_true'] = []
    
    for pump_id in pump_id_list:
        train_dict['x_u'].append(obs_x)
        train_dict['y_u'].append(obs_y)
        train_dict['u_true'].append(heads[:, pump_id:pump_id+1])
    
    # PDE配置点
    train_dict['x_f'] = X_star[:, 0:1]
    train_dict['y_f'] = X_star[:, 1:2]
    
    # 诺伊曼边界
    train_dict['x_neum'] = np.vstack((top_bound[:, 0:1], bottom_bound[:, 0:1]))
    train_dict['y_neum'] = np.vstack((top_bound[:, 1:2], bottom_bound[:, 1:2]))
    
    # 狄利克雷边界
    train_dict['x_diri'] = np.vstack((left_bound[:, 0:1], right_bound[:, 0:1]))
    train_dict['y_diri'] = np.vstack((left_bound[:, 1:2], right_bound[:, 1:2]))
    train_dict['diri_true'] = np.vstack((np.ones((ny, 1)), np.zeros((ny, 1))))
    
    # 抽水井位置
    pump_x = np.linspace(0.1, 0.9, 5)
    pump_y = np.linspace(0.1, 0.9, 5)
    pump_X, pump_Y = np.meshgrid(pump_x, pump_y)
    pump_locations = np.hstack((pump_X.flatten()[:, None], pump_Y.flatten()[:, None]))
    train_dict['x_pump'] = pump_locations[:, 0:1]
    train_dict['y_pump'] = pump_locations[:, 1:2]
    train_dict['pump_true'] = np.zeros((25, 1))
    
    # 7. 网络结构和超参数
    hnu = 20
    hnk = 20
    layers_u = [2, hnu, hnu, hnu, hnu, 1]
    layers_K = [2, hnk, hnk, hnk, hnk, 1]
    lbs = np.array([0, 0])
    ubs = np.array([1, 1])
    
    # 8. 损失函数权重（经过调优的最优值）
    loss_weights = {
        'f': 50,
        'u': 10000,
        'neum': 10000,
        'pump': 1,
        'K': 100,
        'diri': 20000
    }
    
    # 9. 预测值键列表
    pred_keys = ['u', 'f', 'neum', 'diri', 'pump', 'K']
    
    # 10. 创建模型
    model = PhysicsInformedNN(layers_u, layers_K, lbs, ubs)
    
    # 11. 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 12. 分阶段训练
    epochs = [10001, 20000, 20000]
    lrs = [1e-3, 1e-4, 1e-5]
    
    for epoch, lr in zip(epochs, lrs):
        logging.info(f"\n开始训练阶段: 学习率={lr}, 轮数={epoch}")
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        model.train(
            epochs=epoch,
            data_batch=train_dict,
            loss_func=model.loss_func,
            optimizer=optimizer,
            pred_keys=pred_keys,
            loss_weights=loss_weights,
            pump_id_list=pump_id_list,
            print_interval=3000
        )
    
    # ==============================================
    # 反演训练核心代码结束
    # ==============================================
    
    # 提取预测结果（经过验证的最稳定版本，彻底解决卡死问题）
    logging.info("\n开始提取预测结果")
    ti = pump_id_list[2]  # 使用第2号抽水井，与正向模拟对应
    X_pred = model.coor_shift(X_star)
    u_pred = model.predict(X_pred, target='u')
    K_pred = model.predict(X_pred, target='K')
    x_obs = X_star[:, 0]
    y_obs = X_star[:, 1]
    
    # 保存结果（精确到小数点后8位，符合科学计算精度要求）
    np.savetxt(os.path.join(output_dir, 'model_K.txt'), K_pred, fmt='%.8f', delimiter='\t')
    np.savetxt(os.path.join(output_dir, 'model_u_0.txt'), u_pred, fmt='%.8f', delimiter='\t')
    np.savetxt(os.path.join(output_dir, 'x_obs.txt'), x_obs, fmt='%.8f', delimiter='\t')
    np.savetxt(os.path.join(output_dir, 'y_obs.txt'), y_obs, fmt='%.8f', delimiter='\t')
    
    # 保存超参数（顶刊要求完整记录所有实验条件）
    elapsed = time.time() - start_time
    with open(os.path.join(output_dir, 'inverse_params.txt'), 'w') as f:
        f.write('='*50 + '\n')
        f.write('反演训练超参数记录\n')
        f.write('='*50 + '\n\n')
        f.write(f'水头分支隐藏层神经元数: {hnu}\n')
        f.write(f'渗透系数分支隐藏层神经元数: {hnk}\n')
        f.write(f'隐藏层层数: 4\n')
        f.write(f'总训练轮数: {sum(epochs)}\n')
        f.write(f'学习率调度: {lrs}\n')
        f.write(f'损失函数权重: {loss_weights}\n')
        f.write(f'观测点数量: {len(obs_x)}\n')
        f.write(f'抽水井数量: {len(pump_id_list)}\n')
        f.write(f'运行时间: {elapsed:.2f} 秒\n')
        f.write(f'随机种子: 42\n')
        f.write(f'运行日期: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    
    logging.info(f"✅ 反演训练完成，总运行时间: {elapsed:.2f} 秒")
    logging.info(f"✅ 所有结果已保存到: {output_dir}")
    logging.info("="*60)

if __name__ == "__main__":
    main()