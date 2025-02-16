// data.cu

#include "data.h"
#include <iostream>
#include <vector>

// 在 GPU 上分配内存
void allocateDeviceMemory(solVectors &d_data, int n) {
    cudaMalloc((void**)&(d_data.rho), n * sizeof(float));
    cudaMalloc((void**)&(d_data.vx),  n * sizeof(float));
    cudaMalloc((void**)&(d_data.vy),  n * sizeof(float));
    cudaMalloc((void**)&(d_data.p),   n * sizeof(float));
}

// 在 GPU 上释放内存
void freeDeviceMemory(solVectors &d_data) {
    cudaFree(d_data.rho);
    cudaFree(d_data.vx);
    cudaFree(d_data.vy);
    cudaFree(d_data.p);
}

// 初始化数据并复制到 GPU
void initDataAndCopyToGPU(solVectors &d_data, int n) {
    // 在 CPU 上先创建四个向量存放初始值
    std::vector<float> h_rho(n, 1.0f);  // 假设初始密度=1
    std::vector<float> h_vx(n, 0.0f);   // 假设初始速度=0
    std::vector<float> h_vy(n, 0.0f);
    std::vector<float> h_p(n, 1.0f);    // 假设初始压力=1

    // 例如：可以根据位置或某些条件给这些向量赋值
    // 这里只是演示，实际可做更复杂初始化
    for(int i = 0; i < n; i++){
        // 例如：某处做个脉冲，简单演示
        if (i > n/4 && i < n/3) {
            h_rho[i] = 2.0f;
            h_p[i] = 1.5f;
        }
    }

    // 拷贝到 GPU
    cudaMemcpy(d_data.rho, h_rho.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data.vx,  h_vx.data(),  n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data.vy,  h_vy.data(),  n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data.p,   h_p.data(),   n * sizeof(float), cudaMemcpyHostToDevice);
}

// 一个简单的 kernel 示例：每个线程把 p[i] 加 1.0
__global__ void kernelExample(solVectors d_data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        d_data.p[idx] += 1.0f; // 简单操作：压力数组加 1
    }
}
