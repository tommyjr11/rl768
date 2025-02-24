import numpy as np
import matplotlib.pyplot as plt
import glob

# ========== 1. 自动寻找 CSV 文件 ==========
file_list = sorted(glob.glob('data/*.csv'))
if not file_list:
    print("No CSV files found in 'data' directory.")
    exit()

filename = file_list[0]  # 只取找到的第一个 CSV 文件
print(f"Using file: {filename}")

# ========== 2. 读取数据 ==========
data = np.loadtxt(filename, delimiter=',', skiprows=1)
if data.size == 0:
    print(f"No data in file {filename}")
    exit()

# 假设网格是 (nx, ny)
ny, nx4 = data.shape
nx = nx4 // 4
data = data.reshape((ny, nx, 4))

rho = data[..., 0]   # 密度
vx  = data[..., 1]   # x方向速度
vy  = data[..., 2]   # y方向速度
p   = data[..., 3]   # 压力

# ========== 3. 网格坐标 ==========
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# ========== 4. 计算全局压力范围 ==========
p_min, p_max = np.min(p), np.max(p)

# ========== 5. 设定密度等值线 ==========
rho_levels = np.arange(0.16, 1.71 + 0.1, 0.1)  # 0.2 ~ 1.6，每 0.1 取一条线

# ========== 6. 速度矢量场下采样 ==========
skip = max(1, nx // 50)  # 让箭头密度随网格分辨率调整

# ========== 7. 绘制图像 ==========
fig, ax = plt.subplots(figsize=(6, 6))

# (1) 以压力 p 作为背景颜色
c = ax.contourf(X, Y, p, levels=50, cmap='jet', vmin=p_min, vmax=p_max)

# (2) 以密度 rho 作为等值线（黑色线）
ax.contour(X, Y, rho, levels=rho_levels, colors='k', linewidths=0.5)

# (3) 速度矢量场（减少箭头数量）
ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
          vx[::skip, ::skip], vy[::skip, ::skip], 
          color='white', scale=40, width=0.002)

ax.set_title('Final Time Step', fontsize=12)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_aspect('equal', adjustable='box')

# 添加 colorbar
fig.colorbar(c, ax=ax, label='Pressure')

# ========== 8. 保存优化后的高质量图片 ==========
plt.tight_layout()
plt.savefig('final_frame_optimized.png', dpi=1200, bbox_inches='tight')  # dpi=600，避免太大
plt.show()
