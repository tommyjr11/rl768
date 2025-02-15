import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob

# ========== 1. 读取数据 ==========

# 假设所有 CSV 文件命名为 step_*.csv，存放于 data/ 目录
file_list = sorted(glob.glob('data/step_*.csv'))
if not file_list:
    print("No data files found in the 'data' directory.")
    exit()

rho_data_list = []
vx_data_list = []
vy_data_list = []
p_data_list  = []

for filename in file_list:
    with open(filename, 'r') as f:
        # 假设每个文件格式是:
        # x方向上有 nx 个点，y方向上有 ny 个点，每行 (rho, vx, vy, p)
        # skiprows=1 如果文件里有表头，否则设为 skiprows=0
        data = np.loadtxt(f, delimiter=',', skiprows=1)
        if data.size == 0:
            print(f"No data in file {filename}")
            continue
        
        # 根据实际 nx, ny 调整，这里先假设数据是 ny 行，nx*4 列
        # 如果无法确定，可手动设置 nx, ny
        # 例如 nx=400, ny=400
        ny, nx4 = data.shape
        nx = nx4 // 4
        
        data = data.reshape((ny, nx, 4))
        
        rho_data_list.append(data[..., 0])   # 密度
        vx_data_list.append(data[..., 1])    # x方向速度
        vy_data_list.append(data[..., 2])    # y方向速度
        p_data_list.append(data[..., 3])     # 压力

# ========== 2. 网格坐标 (假设物理域为 [0,1]×[0,1]) ==========

ny, nx = rho_data_list[0].shape
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# ========== 3. 设置全局压力范围 (用于 colorbar) ==========

p_min = min(np.min(p_frame) for p_frame in p_data_list)
p_max = max(np.max(p_frame) for p_frame in p_data_list)

# 如果想手动固定范围，比如 [0.16, 1.6]，可自行设置:
# p_min, p_max = 0.0, 1.6

# ========== 4. 定义密度等值线 (32 条, 0.16~1.71, 步长 0.05) ==========

rho_levels = np.arange(0.16, 1.71 + 0.05, 0.05)  # 0.16, 0.21, ..., 1.71

# ========== 5. 设置箭头 (速度矢量) 的下采样步长 ==========

skip = 10  # 根据实际网格大小可做调整

# ========== 6. 创建动画所需的 figure, axes ==========

# figsize=(6,6) 保持绘图区域为正方形大小
fig, ax = plt.subplots(figsize=(6, 6))

# 动画更新函数
def update_plot(frame):
    ax.clear()
    
    # (1) 以压力 p 做背景颜色
    c = ax.contourf(X, Y, p_data_list[frame],
                    levels=50,             # 50个色阶
                    cmap='jet',
                    vmin=p_min, vmax=p_max)
    
    # (2) 以密度 rho 做等值线 (黑色线)
    c_rho = ax.contour(X, Y, rho_data_list[frame],
                       levels=rho_levels,
                       colors='k', linewidths=1)
    
    # (3) 绘制速度矢量场 (箭头)
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              vx_data_list[frame][::skip, ::skip],
              vy_data_list[frame][::skip, ::skip],
              color='white', scale=20, width=0.002)
    
    ax.set_title(f'Time Step: {frame}', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # ======== 关键：让坐标轴比例为 1:1 (正方形显示) ========
    ax.set_aspect('equal', adjustable='box')
    
    return c  # 返回主要的 contourf 对象 (给 FuncAnimation 用)

# 创建 FuncAnimation
ani = animation.FuncAnimation(fig, update_plot,
                              frames=len(rho_data_list),
                              interval=200, blit=False)

# 给背景颜色(压力)加一个 colorbar
sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=p_min, vmax=p_max))
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Pressure')

plt.tight_layout()
plt.show()

# ========== 7. 保存最后一帧图像 ==========

last_frame = len(rho_data_list) - 1  # 获取最后一帧索引

fig_last, ax_last = plt.subplots(figsize=(6, 6))

# (1) 压力背景
c = ax_last.contourf(X, Y, p_data_list[last_frame],
                     levels=50, cmap='jet',
                     vmin=p_min, vmax=p_max)

# (2) 密度等值线
c_rho = ax_last.contour(X, Y, rho_data_list[last_frame],
                        levels=rho_levels,
                        colors='k', linewidths=1)

# (3) 速度矢量场
ax_last.quiver(X[::skip, ::skip], Y[::skip, ::skip],
               vx_data_list[last_frame][::skip, ::skip],
               vy_data_list[last_frame][::skip, ::skip],
               color='white', scale=20, width=0.002)

ax_last.set_title(f'Final Time Step: {last_frame}', fontsize=12)
ax_last.set_xlabel('X')
ax_last.set_ylabel('Y')
ax_last.set_xlim([0, 1])
ax_last.set_ylim([0, 1])

# 同样在最终帧里设置坐标轴等比例
ax_last.set_aspect('equal', adjustable='box')

# colorbar
fig_last.colorbar(sm, ax=ax_last, label='Pressure')

plt.tight_layout()
plt.savefig('final_frame_super.png', dpi=1200, bbox_inches='tight')
plt.close(fig_last)
