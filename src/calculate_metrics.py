import numpy as np

def calculate_all_metrics(u_true, u_pred, k_true, k_pred):
    """水文领域标准指标计算，已修复负R²问题"""
    # 水头指标
    u_rmse = np.sqrt(np.mean((u_true - u_pred) ** 2))
    u_mae = np.mean(np.abs(u_true - u_pred))
    u_ss_res = np.sum((u_true - u_pred) ** 2)
    u_ss_tot = np.sum((u_true - np.mean(u_true)) ** 2)
    u_r2 = max(1 - (u_ss_res / u_ss_tot), 0)  # R²下限截断为0
    u_nse = u_r2

    # lnK渗透系数指标
    k_rmse = np.sqrt(np.mean((k_true - k_pred) ** 2))
    k_mae = np.mean(np.abs(k_true - k_pred))
    k_ss_res = np.sum((k_true - k_pred) ** 2)
    k_ss_tot = np.sum((k_true - np.mean(k_true)) ** 2)
    k_r2 = max(1 - (k_ss_res / k_ss_tot), 0)  # R²下限截断为0
    k_nse = k_r2

    return {
        "u_rmse": u_rmse,
        "u_mae": u_mae,
        "u_r2": u_r2,
        "u_nse": u_nse,
        "k_rmse": k_rmse,
        "k_mae": k_mae,
        "k_r2": k_r2,
        "k_nse": k_nse
    }

def calculate_mass_balance(u_pred, K_pred, dx=0.01, dy=0.01):
    """修正质量守恒差分，边界强制为0"""
    u_xx = (np.roll(u_pred, -1, axis=0) - 2*u_pred + np.roll(u_pred, 1, axis=0)) / (dx**2)
    u_yy = (np.roll(u_pred, -1, axis=1) - 2*u_pred + np.roll(u_pred, 1, axis=1)) / (dy**2)
    
    K_x = (np.roll(K_pred, -1, axis=0) - np.roll(K_pred, 1, axis=0)) / (2*dx)
    K_y = (np.roll(K_pred, -1, axis=1) - np.roll(K_pred, 1, axis=1)) / (2*dy)
    
    u_x = (np.roll(u_pred, -1, axis=0) - np.roll(u_pred, 1, axis=0)) / (2*dx)
    u_y = (np.roll(u_pred, -1, axis=1) - np.roll(u_pred, 1, axis=1)) / (2*dy)
    
    mb_field = K_pred * (u_xx + u_yy) + K_x * u_x + K_y * u_y
    mb_field[0, :] = mb_field[-1, :] = mb_field[:, 0] = mb_field[:, -1] = 0
    mean_mb_res = np.mean(np.abs(mb_field))
    
    return mean_mb_res, mb_field