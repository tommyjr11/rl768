// data.h

#ifndef DATA_H
#define DATA_H
#include <cuda_runtime.h>
#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(-1); \
        } \
    }
// 在此简单规定网格数量（例如一维 N=1000）
const int nx = 100;
const int ny = 100;
const int ghost = 2;
const float C = 0.8;
const float t0 = 0.0;
const float t1 = 0.3;
const float x_width0 = 0.0;
const float x_width1 = 1.0;
const float y_width0 = 0.0;
const float y_width1 = 1.0;
const float dx = (x_width1 - x_width0) / nx;
const float dy = (y_width1 - y_width0) / ny;

// SoA 结构
struct solVectors {
    float *rho;
    float *vx;
    float *vy;
    float *p;
};
// 在 GPU 上分配/释放
void allocateDeviceMemory(solVectors &d_data);
void freeDeviceMemory(solVectors &d_data);
// 初始化并复制到 GPU
void initDataAndCopyToGPU(solVectors &d_data);
// 使用 GPU 计算网格内的最大速度
float getmaxspeedGPU(const solVectors &d_data, float r);
// 计算时间步长 = C * min(dx, dy) / maxSpeed
float getdtGPU(const solVectors &d_data, float r);
// 设置边界条件
void applyBoundaryConditions(solVectors &d_u);

#endif // DATA_H
