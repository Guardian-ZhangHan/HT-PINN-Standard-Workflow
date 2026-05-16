"""
通用工具函数模块
包含随机种子固定、路径处理、日志配置等通用功能
所有顶刊级科研代码的标准配置
"""
import os
import sys
import random
import logging
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    全局固定所有随机种子，确保100%可复现
    参考：https://pytorch.org/docs/stable/notes/randomness.html
    
    Args:
        seed: 随机种子值，默认42（科研界通用基准种子）
    """
    # Python内置随机数
    random.seed(seed)
    
    # Numpy随机数
    np.random.seed(seed)
    
    # PyTorch CPU随机数
    torch.manual_seed(seed)
    
    # PyTorch GPU随机数（如果有GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 禁用cudnn基准模式，确保卷积操作确定性
    torch.backends.cudnn.benchmark = False
    
    # 启用cudnn确定性模式
    torch.backends.cudnn.deterministic = True
    
    # 设置PyTorch确定性算法
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    
    logging.info(f"✅ 所有随机种子已固定为: {seed}")

def setup_logging(log_file: str = None) -> None:
    """
    配置标准日志系统，同时输出到控制台和文件
    替代print语句，方便调试和永久记录实验过程
    """
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='w'))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )

def ensure_dir(dir_path: str) -> str:
    """
    确保目录存在，如果不存在则创建
    避免因目录不存在导致的保存错误
    """
    dir_path = os.path.abspath(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path