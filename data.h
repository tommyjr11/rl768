// data.h
#ifndef DATA_H
#define DATA_H
#include <cuda_runtime.h>
#include <vector>
#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(-1); \
        } \
    }

#define BDIMX_X 64
#define BDIMX_Y 6
#define BDIMY_X 6
#define BDIMY_Y 64
#define nx_point 800
#define ny_point 800
#define SHARE_X_GRID_X (1+(nx_point + 4 - BDIMX_X)/(BDIMX_X - 4))
#define SHARE_X_GRID_Y ((ny_point + 4 + BDIMX_Y - 1) / BDIMX_Y)
#define SHARE_Y_GRID_Y (1+(ny_point + 4 - BDIMY_Y)/(BDIMY_Y - 4))
#define SHARE_Y_GRID_X ((nx_point + 4 + BDIMY_X - 1) / BDIMY_X)
const int nx = nx_point;
const int ny = ny_point;
const int ghost = 2;
const double C = 0.8;
const double t0 = 0.0;
const double t1 = 0.3;
const double x_width0 = 0.0;
const double x_width1 = 1.0;
const double y_width0 = 0.0;
const double y_width1 = 1.0;
const double dx = (x_width1 - x_width0) / nx;
const double dy = (y_width1 - y_width0) / ny;

// SoA 结构
struct solVectors {
    double *rho;
    double *vx;
    double *vy;
    double *p;
};
// 在 GPU 上分配/释放
void allocateDeviceMemory(solVectors &d_data_pri, solVectors &d_data_con);
void freeDeviceMemory(solVectors &d_data_pri, solVectors &d_data_con);
// 初始化并复制到 GPU
void initDataAndCopyToGPU(solVectors &d_data_pri,solVectors d_data_con);
// 使用 GPU 计算网格内的最大速度
double getmaxspeedGPU(const solVectors &d_data_pri, double r);
// 计算时间步长 = C * min(dx, dy) / maxSpeed
double getdtGPU(const solVectors &d_data_pri, double r);
// 设置边界条件
void applyBoundaryConditions(solVectors &d_u);
// half time 
void computeHalftime(
    const solVectors &d_data_con,
    solVectors &d_half_uL,
    solVectors &d_half_uR,
    double dt,
    int choice
);
void computeSLICFlux(
    const solVectors &d_half_uL,
    const solVectors &d_half_uR,
    solVectors &d_SLIC_flux, 
    double dt,
    int choice 
);
void updateSolution(
    solVectors &d_data_con,
    const solVectors &d_SLIC_flux,
    double dt,
    int choice
);
void freeDeviceMemory2(solVectors &d_half_uL, solVectors &d_half_uR, solVectors &d_SLIC_flux);
void list_con2pri(
    solVectors &d_data_con,
    solVectors &d_data_pri
);
void store_data(const std::vector<double> rho, const std::vector<double> vx, const std::vector<double> vy, const std::vector<double> p, const double t, int step);
void launchUpdateSLICKernel(solVectors &d_data_con, double dt);
#endif 
