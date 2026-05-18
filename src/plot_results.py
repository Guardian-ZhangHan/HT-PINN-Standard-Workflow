import numpy as np
import matplotlib.pyplot as plt
import os

# 顶刊标准绘图风格（WRR/JHM通用）
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 读取训练输出文件（路径与训练代码完全一致）
lnK_pred = np.loadtxt("../model_coeff/model_K.txt")
u_pred = np.loadtxt("../model_coeff/model_u.txt")
lnK_true = np.loadtxt("../model_coeff/true_K.txt")
u_true = np.tile(np.linspace(1, 0, 100), (100, 1))

# 自动创建输出目录
os.makedirs("../results/figures", exist_ok=True)

# ===================== 图1：lnK渗透系数场对比（核心创新成果图） =====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

im1 = ax1.imshow(lnK_true, cmap='jet', origin='lower', extent=[0, 1, 0, 1])
ax1.set_title('True lnK Field', fontweight='bold', pad=10)
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('lnK (m/d)', labelpad=10)

im2 = ax2.imshow(lnK_pred, cmap='jet', origin='lower', extent=[0, 1, 0, 1])
ax2.set_title('Predicted lnK Field', fontweight='bold', pad=10)
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('lnK (m/d)', labelpad=10)

plt.tight_layout()
plt.savefig("../results/figures/lnK_comparison.pdf")
plt.close()

# ===================== 图2：水头场对比 =====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

im1 = ax1.imshow(u_true, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
ax1.set_title('True Hydraulic Head', fontweight='bold', pad=10)
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Head (m)', labelpad=10)

im2 = ax2.imshow(u_pred, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
ax2.set_title('Predicted Hydraulic Head', fontweight='bold', pad=10)
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Head (m)', labelpad=10)

plt.tight_layout()
plt.savefig("../results/figures/head_comparison.pdf")
plt.close()

# ===================== 图3：lnK预测误差分布 =====================
error = lnK_pred - lnK_true
fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(error, cmap='coolwarm', origin='lower', extent=[0, 1, 0, 1])
ax.set_title('lnK Prediction Error', fontweight='bold', pad=10)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Error (lnK)', labelpad=10)

plt.tight_layout()
plt.savefig("../results/figures/lnK_error.pdf")
plt.close()

print("="*70)
print("✅ All figures generated successfully!")
print("📁 Saved to: ../results/figures/")
print("📄 Files:")
print("   - lnK_comparison.pdf (核心图)")
print("   - head_comparison.pdf")
print("   - lnK_error.pdf")
print("="*70)
print("All figures are high-resolution PDF format, ready for journal submission.")