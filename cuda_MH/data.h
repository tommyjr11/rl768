// data.h

#ifndef DATA_H
#define DATA_H

#include <cuda_runtime.h>

// 网格数量
const int N = 1000;  // 这里只是举例，实际可根据需要

// SoA 布局的数据结构
struct solVectors {
    float *rho;
    float *vx;
    float *vy;
    float *p;
};

// 在 GPU 上为 solVectors 分配/释放内存
void allocateDeviceMemory(solVectors &d_data, int n);
void freeDeviceMemory(solVectors &d_data);

// 初始化数据并复制到 GPU
void initDataAndCopyToGPU(solVectors &d_data, int n);

// 核函数示例：在 GPU 上做某些操作（如简单的加法）
__global__ void kernelExample(solVectors d_data, int n);

#endif // DATA_H
