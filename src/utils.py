import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    """固定所有随机种子，100%可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 强制单线程运行，消除多线程随机误差
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

def save_checkpoint(model, optimizer, scheduler, epoch, loss_total, path):
    """保存完整检查点"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "loss_total": loss_total.item()
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, scheduler, path, load_optimizer=True):
    """加载检查点"""
    checkpoint = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if load_optimizer and optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if load_optimizer and scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["loss_total"]
    return model, optimizer, scheduler, start_epoch, best_loss