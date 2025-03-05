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
print(f"Data shape: {data.shape}")
nx = nx4 // 4
print(f"Grid size: {nx} x {ny}")
data = data.reshape((ny, nx, 4))

rho = data[..., 0]   # 密度
vx  = data[..., 1]   # x方向速度
vy  = data[..., 2]   # y方向速度
p   = data[..., 3]   # 压力
print(f"Min pressure: {np.min(rho)}, Max pressure: {np.max(rho)}")
# ========== 3. 网格坐标 ==========
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# ========== 4. 计算全局压力范围 ==========
p_min, p_max = np.min(p), np.max(p)

# ========== 5. 设定密度等值线 =====
# =====
rho_levels = np.arange(np.min(rho),  np.max(rho), 0.05)  
# rho_levels = np.arange(0.0,  1.7, 0.12)  

# ========== 6. 速度矢量场下采样 ==========
skip = max(1, nx // 50)  # 让箭头密度随网格分辨率调整

# ========== 7. 绘制图像 ==========
# 计算合适的图片大小，保持网格比例
fig_width = 10  # 你可以调整这个值
fig_height = fig_width / (nx / ny)  # 保持 x/y 比例

fig, ax = plt.subplots(figsize=(fig_width, fig_height))  # 调整图像大小

# (1) 以压力 p 作为背景颜色
c = ax.contourf(X, Y, p, levels=50, cmap='jet', vmin=p_min, vmax=p_max)

# (2) 以密度 rho 作为等值线（黑色线）
ax.contour(X, Y, rho, levels=rho_levels, colors='k', linewidths=0.4)

# # (3) 速度矢量场（减少箭头数量）
# ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
#           vx[::skip, ::skip], vy[::skip, ::skip], 
#           color='white', scale=40, width=0.002)

ax.set_title('Final Time Step', fontsize=12)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_aspect(ny / nx)

# 添加 colorbar
fig.colorbar(c, ax=ax, label='Pressure')

# ========== 8. 保存优化后的高质量图片 ==========
plt.tight_layout()
# name 是 nx加ny的值加filename不要.csv的值
name = "picture_6.2_1.2/"+"nx: " + str(nx) + " ny: " + str(ny) + filename[5:-4] + ".png"
print(name)
plt.savefig(name, dpi=1200, bbox_inches='tight')  # dpi=600，避免太大
plt.show()
