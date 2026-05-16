"""
HT-PINN 模型定义模块
双分支物理信息神经网络，用于地下水渗透系数反演
参考：[原始HT-PINN论文完整引用]
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm

from utils import set_seed

# 全局固定随机种子（必须放在最开头）
set_seed(42)

class PhysicsInformedNN(nn.Module):
    """
    双分支物理信息神经网络类
    
    网络结构：
    - 水头分支(u): 输入(x,y)，输出水头值u(x,y)
    - 渗透系数分支(K): 输入(x,y)，输出渗透系数对数值lnK(x,y)
    
    物理约束：稳态地下水流动方程 ∇·(K∇u) = 0
    """
    
    def __init__(
        self,
        layers_u: List[int],
        layers_K: List[int],
        lbs: np.ndarray,
        ubs: np.ndarray,
        activation: str = 'tanh',
        output_activation: str = 'linear'
    ):
        """
        初始化PINN模型
        
        Args:
            layers_u: 水头分支网络结构，例如 [2, 20, 20, 20, 20, 1]
            layers_K: 渗透系数分支网络结构，例如 [2, 20, 20, 20, 20, 1]
            lbs: 输入坐标的最小值，形状为 (2,)
            ubs: 输入坐标的最大值，形状为 (2,)
            activation: 隐藏层激活函数，默认'tanh'
            output_activation: 输出层激活函数，默认'linear'
        """
        super(PhysicsInformedNN, self).__init__()
        
        self.lbs = torch.tensor(lbs, dtype=torch.float32)
        self.ubs = torch.tensor(ubs, dtype=torch.float32)
        
        # 初始化水头分支
        self.u_net = self._build_mlp(layers_u, activation, output_activation)
        
        # 初始化渗透系数分支
        self.K_net = self._build_mlp(layers_K, activation, output_activation)
        
        # 初始化权重（Xavier均匀初始化，有明确文献引用）
        self._initialize_weights()
        
        # 记录网络结构
        self.layers_u = layers_u
        self.layers_K = layers_K
        self.activation = activation
        self.output_activation = output_activation
        
        logging.info("✅ PINN模型初始化完成")
        logging.info(f"   水头分支结构: {layers_u}")
        logging.info(f"   渗透系数分支结构: {layers_K}")
    
    def _build_mlp(
        self,
        layers: List[int],
        activation: str,
        output_activation: str
    ) -> nn.Sequential:
        """
        构建多层感知机(MLP)
        解耦网络构建逻辑，方便后续修改网络结构
        """
        mlp = nn.Sequential()
        
        # 隐藏层
        for i in range(len(layers) - 2):
            mlp.add_module(f'linear_{i}', nn.Linear(layers[i], layers[i+1]))
            if activation == 'tanh':
                mlp.add_module(f'tanh_{i}', nn.Tanh())
            elif activation == 'relu':
                mlp.add_module(f'relu_{i}', nn.ReLU())
            elif activation == 'sigmoid':
                mlp.add_module(f'sigmoid_{i}', nn.Sigmoid())
            else:
                raise ValueError(f"不支持的激活函数: {activation}")
        
        # 输出层
        mlp.add_module('output', nn.Linear(layers[-2], layers[-1]))
        if output_activation != 'linear':
            if output_activation == 'tanh':
                mlp.add_module('output_tanh', nn.Tanh())
            elif output_activation == 'relu':
                mlp.add_module('output_relu', nn.ReLU())
        
        return mlp
    
    def _initialize_weights(self) -> None:
        """
        使用Xavier均匀初始化网络权重
        参考：Glorot et al. (2010) Understanding the difficulty of training deep feedforward neural networks
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def coor_shift(self, x: np.ndarray) -> torch.Tensor:
        """
        将输入坐标归一化到[-1, 1]区间
        提高神经网络训练稳定性
        """
        x = torch.tensor(x, dtype=torch.float32)
        return 2.0 * (x - self.lbs) / (self.ubs - self.lbs) - 1.0
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        返回水头和渗透系数的预测值
        """
        u = self.u_net(x)
        K = self.K_net(x)
        return u, K
    
    def predict(
        self,
        x: torch.Tensor,
        pid: int = None,
        target: str = 'all'
    ) -> np.ndarray:
        """
        预测函数，用于生成最终结果
        自动切换到评估模式，禁用梯度计算
        """
        self.eval()
        with torch.no_grad():
            u, K = self.forward(x)
            
            if target == 'u':
                return u.cpu().numpy().flatten()
            elif target == 'K':
                return K.cpu().numpy().flatten()
            elif target == 'all':
                return u.cpu().numpy().flatten(), K.cpu().numpy().flatten()
            else:
                raise ValueError(f"不支持的预测目标: {target}")
    
    def loss_func(
        self,
        pred_dict: Dict[str, torch.Tensor],
        true_dict: Dict[str, torch.Tensor],
        pump_id: int,
        loss_weights: Dict[str, float]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失函数
        包含数据损失、PDE损失、诺伊曼边界损失、狄利克雷边界损失和抽水井损失
        
        Args:
            pred_dict: 模型预测值字典
            true_dict: 真实值字典
            pump_id: 当前抽水井ID
            loss_weights: 各损失项的权重字典
        
        Returns:
            total_loss: 总损失值
            loss_dict: 各分项损失值字典
        """
        # 数据损失（观测点水头损失）
        loss_u = torch.mean((pred_dict['u'] - true_dict['u'])**2)
        
        # PDE损失（地下水流动方程损失）
        loss_f = torch.mean(pred_dict['f']**2)
        
        # 诺伊曼边界损失（流量边界）
        loss_neum = torch.mean(pred_dict['neum']**2)
        
        # 狄利克雷边界损失（水头边界）
        loss_diri = torch.mean(pred_dict['diri']**2)
        
        # 抽水井损失
        loss_pump = torch.mean(pred_dict['pump']**2)
        
        # 渗透系数正则化损失
        loss_K = torch.mean(pred_dict['K']**2)
        
        # 加权总损失
        total_loss = (
            loss_weights['u'] * loss_u +
            loss_weights['f'] * loss_f +
            loss_weights['neum'] * loss_neum +
            loss_weights['diri'] * loss_diri +
            loss_weights['pump'] * loss_pump +
            loss_weights['K'] * loss_K
        )
        
        # 记录各分项损失
        loss_dict = {
            'total': total_loss.item(),
            'u': loss_u.item(),
            'f': loss_f.item(),
            'neum': loss_neum.item(),
            'diri': loss_diri.item(),
            'pump': loss_pump.item(),
            'K': loss_K.item()
        }
        
        return total_loss, loss_dict
    
    def train(
        self,
        epochs: int,
        data_batch: Dict[str, np.ndarray],
        loss_func,
        optimizer,
        pred_keys: List[str],
        loss_weights: Dict[str, float],
        pump_id_list: List[int],
        print_interval: int = 3000
    ) -> None:
        """
        模型训练函数
        包含完整的训练循环、进度显示和日志记录
        
        Args:
            epochs: 训练轮数
            data_batch: 训练数据批次字典
            loss_func: 损失函数
            optimizer: 优化器
            pred_keys: 预测值键列表
            loss_weights: 各损失项的权重字典
            pump_id_list: 抽水井ID列表
            print_interval: 日志打印间隔，默认3000轮
        """
        logging.info(f"开始训练，总轮数: {epochs}")
        logging.info(f"损失权重: {loss_weights}")
        
        # 切换到训练模式
        self.train()
        
        # 训练循环
        for epoch in tqdm(range(epochs), desc="训练进度"):
            # 清零梯度
            optimizer.zero_grad()
            
            total_loss = 0.0
            loss_dict_total = {}
            
            # 遍历所有抽水井
            for pump_id in pump_id_list:
                # 解包训练数据
                train_dict = self.unzip_train_dict(data_batch, pred_keys, pump_id)
                
                # 前向传播
                pred_dict = self.forward(train_dict['x_tensors'], train_dict['y_tensors'],
                                        self.weights_u[pump_id], self.biases_u[pump_id],
                                        keys=pred_keys)
                
                # 计算损失
                loss, loss_dict = loss_func(pred_dict, train_dict['true_dict'], pump_id, loss_weights)
                
                # 累加损失
                total_loss += loss
                
                # 累加各分项损失
                for k, v in loss_dict.items():
                    if k not in loss_dict_total:
                        loss_dict_total[k] = 0.0
                    loss_dict_total[k] += v
            
            # 反向传播
            total_loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 打印日志
            if (epoch + 1) % print_interval == 0:
                # 计算平均损失
                avg_loss_dict = {k: v / len(pump_id_list) for k, v in loss_dict_total.items()}
                
                logging.info(f"Iter # {epoch+1}, Loss: {avg_loss_dict['total']:.6f}")
                logging.info(f"  u: {avg_loss_dict['u']:.6f}, f: {avg_loss_dict['f']:.6f}, "
                            f"neum: {avg_loss_dict['neum']:.6f}, pump: {avg_loss_dict['pump']:.6f}, "
                            f"diri: {avg_loss_dict['diri']:.6f}")
        
        logging.info("✅ 训练完成")
    
    def unzip_train_dict(
        self,
        train_dict: Dict[str, np.ndarray],
        keys: List[str],
        pump_id: int
    ) -> Dict[str, torch.Tensor]:
        """
        解包训练数据字典
        将numpy数组转换为torch张量，并移动到正确的设备
        
        Args:
            train_dict: 训练数据字典
            keys: 需要提取的键列表
            pump_id: 当前抽水井ID
        
        Returns:
            解包后的训练数据字典
        """
        x_tensors = []
        y_tensors = []
        true_dict = {}
        
        for key in keys:
            if key == 'u':
                x = train_dict[f'x_{key}'][pump_id]
                y = train_dict[f'y_{key}'][pump_id]
                true_val = train_dict[f'{key}_true'][pump_id]
                
                x_tensors.append(torch.tensor(x, dtype=torch.float32, requires_grad=True))
                y_tensors.append(torch.tensor(y, dtype=torch.float32, requires_grad=True))
                true_dict[key] = torch.tensor(true_val, dtype=torch.float32)
            else:
                x = train_dict[f'x_{key}']
                y = train_dict[f'y_{key}']
                
                x_tensors.append(torch.tensor(x, dtype=torch.float32, requires_grad=True))
                y_tensors.append(torch.tensor(y, dtype=torch.float32, requires_grad=True))
        
        return {
            'x_tensors': x_tensors,
            'y_tensors': y_tensors,
            'true_dict': true_dict
        }
    
    def forward(
        self,
        x_tensors: List[torch.Tensor],
        y_tensors: List[torch.Tensor],
        weights_u: List[torch.Tensor],
        biases_u: List[torch.Tensor],
        keys: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        多任务前向传播函数
        计算所有预测值和PDE残差
        
        Args:
            x_tensors: x坐标张量列表
            y_tensors: y坐标张量列表
            weights_u: 水头分支权重列表
            biases_u: 水头分支偏置列表
            keys: 需要计算的预测值键列表
        
        Returns:
            预测值和残差字典
        """
        preds = {}
        
        for i, key in enumerate(keys):
            x = x_tensors[i]
            y = y_tensors[i]
            
            # 归一化坐标
            X = torch.cat([x, y], dim=1)
            X = 2.0 * (X - self.lbs) / (self.ubs - self.lbs) - 1.0
            
            if key == 'u':
                # 水头预测
                u = self.u_net(X)
                preds['u'] = u
                
                # 计算梯度用于PDE
                u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
                u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
                u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
                u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
                
                # 渗透系数预测
                K = torch.exp(self.K_net(X))
                K_x = torch.autograd.grad(K.sum(), x, create_graph=True)[0]
                K_y = torch.autograd.grad(K.sum(), y, create_graph=True)[0]
                
                # PDE残差：∇·(K∇u) = K∇²u + ∇K·∇u = 0
                f = K * (u_xx + u_yy) + K_x * u_x + K_y * u_y
                preds['f'] = f
                
            elif key == 'neum':
                # 诺伊曼边界残差
                u = self.u_net(X)
                u_n = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
                preds['neum'] = u_n
                
            elif key == 'diri':
                # 狄利克雷边界残差
                u = self.u_net(X)
                preds['diri'] = u
                
            elif key == 'pump':
                # 抽水井残差
                u = self.u_net(X)
                preds['pump'] = u
                
            elif key == 'K':
                # 渗透系数预测
                K = self.K_net(X)
                preds['K'] = K
        
        return preds