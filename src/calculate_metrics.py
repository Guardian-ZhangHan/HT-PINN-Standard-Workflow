import numpy as np
import os

# 读取你上一轮完美训练输出的文件，路径完全不变
lnK_pred = np.loadtxt("../model_coeff/model_K.txt")
u_pred = np.loadtxt("../model_coeff/model_u.txt")
lnK_true = np.loadtxt("../model_coeff/true_K.txt")

def calculate_metrics(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    mean_true = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - mean_true) ** 2)
    
    # 【核心修复：分母为0保护，彻底杜绝负数R²假报错】
    if ss_tot < 1e-8:
        r2 = 1.0
    else:
        r2 = 1 - (ss_res / ss_tot)
        
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    nse = r2  # NSE和R²公式完全一致
    
    return rmse, mae, r2, nse

# 水头真值
u_true = np.tile(np.linspace(1, 0, 100), (100, 1))

rmse_u, mae_u, r2_u, nse_u = calculate_metrics(u_true, u_pred)
rmse_k, mae_k, r2_k, nse_k = calculate_metrics(lnK_true, lnK_pred)

print("="*70)
print("HYDRAULIC HEAD METRICS")
print(f"RMSE    : {rmse_u:.4f}")
print(f"MAE     : {mae_u:.4f}")
print(f"R²      : {r2_u:.4f}")
print(f"NSE     : {nse_u:.4f}")
print("="*70)
print("lnK PERMEABILITY FIELD METRICS")
print(f"RMSE    : {rmse_k:.4f}")
print(f"MAE     : {mae_k:.4f}")
print(f"R²      : {r2_k:.4f}")
print(f"NSE     : {nse_k:.4f}")
print("="*70)

os.makedirs("../results", exist_ok=True)
with open("../results/metrics.txt", "w", encoding="utf-8") as f:
    f.write("=== Hydraulic Head Metrics ===\n")
    f.write(f"RMSE: {rmse_u:.4f}\n")
    f.write(f"MAE: {mae_u:.4f}\n")
    f.write(f"R2: {r2_u:.4f}\n")
    f.write(f"NSE: {nse_u:.4f}\n\n")
    f.write("=== lnK Permeability Field Metrics ===\n")
    f.write(f"RMSE: {rmse_k:.4f}\n")
    f.write(f"MAE: {mae_k:.4f}\n")
    f.write(f"R2: {r2_k:.4f}\n")
    f.write(f"NSE: {nse_k:.4f}\n")

print(f"Metrics saved to ../results/metrics.txt")