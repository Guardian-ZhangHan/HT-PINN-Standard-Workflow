"""
正向模拟脚本
生成真实地下水水头场和渗透系数场
参考：[原始HT-PINN论文完整引用]
"""
import os
import time
import numpy as np
import scipy.io as sio

from utils import set_seed, setup_logging, ensure_dir
from model import PhysicsInformedNN

def main():
    # 初始化日志（同时输出到控制台和文件）
    setup_logging('../logs/forward.log')
    logging.info("="*60)
    logging.info("开始运行正向模拟")
    logging.info("="*60)
    
    # 固定随机种子
    set_seed(42)
    
    # 确保所有输出目录存在
    output_dir = ensure_dir('../model_coeff')
    log_dir = ensure_dir('../logs')
    
    start_time = time.time()
    
    # ==============================================
    # 完整正向模拟核心代码（从HT_PINN_forward.ipynb提取）
    # ==============================================
    # 1. 加载合成数据
    try:
        data = sio.loadmat('../data/HT_synthetic.mat')
    except FileNotFoundError:
        logging.error("❌ 数据文件不存在: ../data/HT_synthetic.mat")
        logging.error("请确保数据文件放在正确的位置")
        return
    
    # 2. 提取真实渗透系数场
    logK = data['logK']
    K = np.exp(logK)
    nx, ny = logK.shape
    
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
    
    # 5. 抽水井位置（25个均匀分布的抽水井）
    pump_x = np.linspace(0.1, 0.9, 5)
    pump_y = np.linspace(0.1, 0.9, 5)
    pump_X, pump_Y = np.meshgrid(pump_x, pump_y)
    pump_locations = np.hstack((pump_X.flatten()[:, None], pump_Y.flatten()[:, None]))
    num_pumps = len(pump_locations)
    
    # 6. 初始化水头矩阵
    heads = np.zeros((nx*ny, num_pumps))
    
    # 7. 对每个抽水井运行正向模拟
    logging.info(f"开始运行{num_pumps}个抽水井的正向模拟")
    for i in range(num_pumps):
        logging.info(f"正在运行第 {i+1}/{num_pumps} 个抽水井")
        
        # 当前抽水井位置
        pump_x_i = pump_locations[i, 0]
        pump_y_i = pump_locations[i, 1]
        
        # 创建PINN正向模型
        layers = [2, 20, 20, 20, 20, 1]
        lbs = np.array([0, 0])
        ubs = np.array([1, 1])
        model = PhysicsInformedNN(layers, layers, lbs, ubs)
        
        # 训练正向模型
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        epochs = 20000
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 计算边界损失和PDE损失
            # 左边界损失
            x_left = torch.tensor(left_bound[:, 0:1], dtype=torch.float32, requires_grad=True)
            y_left = torch.tensor(left_bound[:, 1:2], dtype=torch.float32, requires_grad=True)
            X_left = torch.cat([x_left, y_left], dim=1)
            X_left = 2.0 * (X_left - lbs) / (ubs - lbs) - 1.0
            u_left = model.u_net(X_left)
            loss_left = torch.mean((u_left - 1.0)**2)
            
            # 右边界损失
            x_right = torch.tensor(right_bound[:, 0:1], dtype=torch.float32, requires_grad=True)
            y_right = torch.tensor(right_bound[:, 1:2], dtype=torch.float32, requires_grad=True)
            X_right = torch.cat([x_right, y_right], dim=1)
            X_right = 2.0 * (X_right - lbs) / (ubs - lbs) - 1.0
            u_right = model.u_net(X_right)
            loss_right = torch.mean((u_right - 0.0)**2)
            
            # 上下边界损失
            x_top = torch.tensor(top_bound[:, 0:1], dtype=torch.float32, requires_grad=True)
            y_top = torch.tensor(top_bound[:, 1:2], dtype=torch.float32, requires_grad=True)
            X_top = torch.cat([x_top, y_top], dim=1)
            X_top = 2.0 * (X_top - lbs) / (ubs - lbs) - 1.0
            u_top = model.u_net(X_top)
            u_top_y = torch.autograd.grad(u_top.sum(), y_top, create_graph=True)[0]
            loss_top = torch.mean(u_top_y**2)
            
            x_bottom = torch.tensor(bottom_bound[:, 0:1], dtype=torch.float32, requires_grad=True)
            y_bottom = torch.tensor(bottom_bound[:, 1:2], dtype=torch.float32, requires_grad=True)
            X_bottom = torch.cat([x_bottom, y_bottom], dim=1)
            X_bottom = 2.0 * (X_bottom - lbs) / (ubs - lbs) - 1.0
            u_bottom = model.u_net(X_bottom)
            u_bottom_y = torch.autograd.grad(u_bottom.sum(), y_bottom, create_graph=True)[0]
            loss_bottom = torch.mean(u_bottom_y**2)
            
            # PDE损失
            x_pde = torch.tensor(X_star[:, 0:1], dtype=torch.float32, requires_grad=True)
            y_pde = torch.tensor(X_star[:, 1:2], dtype=torch.float32, requires_grad=True)
            X_pde = torch.cat([x_pde, y_pde], dim=1)
            X_pde = 2.0 * (X_pde - lbs) / (ubs - lbs) - 1.0
            u = model.u_net(X_pde)
            
            u_x = torch.autograd.grad(u.sum(), x_pde, create_graph=True)[0]
            u_y = torch.autograd.grad(u.sum(), y_pde, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x.sum(), x_pde, create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y.sum(), y_pde, create_graph=True)[0]
            
            # 计算K值（真实值）
            K_interp = np.interp(X_star[:, 0], x, K)
            K_interp = np.interp(X_star[:, 1], y, K_interp.T).T
            K_tensor = torch.tensor(K_interp.flatten()[:, None], dtype=torch.float32)
            
            f = K_tensor * (u_xx + u_yy)
            loss_f = torch.mean(f**2)
            
            # 抽水井损失
            pump_x_tensor = torch.tensor([[pump_x_i]], dtype=torch.float32, requires_grad=True)
            pump_y_tensor = torch.tensor([[pump_y_i]], dtype=torch.float32, requires_grad=True)
            X_pump = torch.cat([pump_x_tensor, pump_y_tensor], dim=1)
            X_pump = 2.0 * (X_pump - lbs) / (ubs - lbs) - 1.0
            u_pump = model.u_net(X_pump)
            loss_pump = torch.mean((u_pump - 0.0)**2)  # 抽水井处水头为0
            
            # 总损失
            total_loss = 10000*loss_left + 10000*loss_right + 10000*loss_top + 10000*loss_bottom + 50*loss_f + 1*loss_pump
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5000 == 0:
                logging.info(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.6f}")
        
        # 预测整个计算域的水头
        X_pred = model.coor_shift(X_star)
        u_pred = model.predict(X_pred, target='u')
        heads[:, i] = u_pred
    
    # ==============================================
    # 正向模拟核心代码结束
    # ==============================================
    
    # 保存结果（精确到小数点后8位，符合科学计算精度要求）
    np.savetxt(os.path.join(output_dir, 'K_true.txt'), logK, fmt='%.8f', delimiter='\t')
    np.savetxt(os.path.join(output_dir, 'u_true.txt'), heads[:, 12], fmt='%.8f', delimiter='\t')  # 选择第12号抽水井
    np.savetxt(os.path.join(output_dir, 'X.txt'), X, fmt='%.8f', delimiter='\t')
    np.savetxt(os.path.join(output_dir, 'Y.txt'), Y, fmt='%.8f', delimiter='\t')
    
    # 保存超参数（顶刊要求完整记录所有实验条件）
    elapsed = time.time() - start_time
    with open(os.path.join(output_dir, 'forward_params.txt'), 'w') as f:
        f.write('='*50 + '\n')
        f.write('正向模拟超参数记录\n')
        f.write('='*50 + '\n\n')
        f.write(f'网格大小: {nx}x{ny}\n')
        f.write(f'计算域大小: 1m x 1m\n')
        f.write(f'抽水井数量: {num_pumps}\n')
        f.write(f'选择的抽水井索引: 12\n')
        f.write(f'正向模型网络结构: {layers}\n')
        f.write(f'训练轮数: {epochs}\n')
        f.write(f'初始学习率: 1e-3\n')
        f.write(f'运行时间: {elapsed:.2f} 秒\n')
        f.write(f'随机种子: 42\n')
        f.write(f'运行日期: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    
    logging.info(f"✅ 正向模拟完成，总运行时间: {elapsed:.2f} 秒")
    logging.info(f"✅ 所有结果已保存到: {output_dir}")
    logging.info("="*60)

if __name__ == "__main__":
    main()