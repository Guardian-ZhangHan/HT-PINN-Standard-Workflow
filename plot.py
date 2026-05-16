import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 全部读硬盘文件，和Jupyter完全无关
K_true = np.loadtxt('./model_coeff/K_true.txt')
u_true = np.loadtxt('./model_coeff/u_true.txt')
X = np.loadtxt('./model_coeff/X.txt')
Y = np.loadtxt('./model_coeff/Y.txt')
K_pred = np.loadtxt('./model_coeff/model_K.txt')
u_pred = np.loadtxt('./model_coeff/model_u_0.txt')
x_obs = np.loadtxt('./model_coeff/x_obs.txt')
y_obs = np.loadtxt('./model_coeff/y_obs.txt')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.dpi'] = 300

fig, axes = plt.subplots(2,2,figsize=(10,8))

axes[0,0].pcolormesh(X,Y,K_true.reshape(X.shape),cmap='jet',vmin=-2,vmax=1.5)
axes[0,0].set_title('(a) True ln$K$ field',fontweight='bold')
axes[0,0].set_xlabel('x (m)')
axes[0,0].set_ylabel('y (m)')

axes[0,1].pcolormesh(X,Y,K_pred.reshape(X.shape),cmap='jet',vmin=-2,vmax=1.5)
axes[0,1].scatter(x_obs,y_obs,c='black',s=20,marker='^',edgecolors='white',linewidths=0.5)
axes[0,1].set_title('(b) Inverted ln$K$ field',fontweight='bold')
axes[0,1].set_xlabel('x (m)')
axes[0,1].set_ylabel('y (m)')

axes[1,0].pcolormesh(X,Y,u_true.reshape(X.shape),cmap='viridis')
axes[1,0].scatter(0,0,c='red',s=60,marker='s',edgecolors='white',linewidths=1)
axes[1,0].set_title('(c) True hydraulic head',fontweight='bold')
axes[1,0].set_xlabel('x (m)')
axes[1,0].set_ylabel('y (m)')

axes[1,1].pcolormesh(X,Y,u_pred.reshape(X.shape),cmap='viridis')
axes[1,1].set_title('(d) Predicted hydraulic head',fontweight='bold')
axes[1,1].set_xlabel('x (m)')
axes[1,1].set_ylabel('y (m)')

plt.tight_layout()
plt.savefig('./最终对比图.png',dpi=300,bbox_inches='tight')
plt.savefig('./最终对比图.pdf',bbox_inches='tight')
plt.show()
print("✅ 对比图生成成功！PNG和PDF已保存到当前文件夹")