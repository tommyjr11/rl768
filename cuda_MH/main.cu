// main.cu

#include <iostream>
#include "data.h"

int main() {
    // 1. 在 GPU 上分配内存
    solVectors d_data;
    allocateDeviceMemory(d_data, N);

    // 2. 初始化并复制初值到 GPU
    initDataAndCopyToGPU(d_data, N);

    // 3. 启动 kernel 做一些测试
    int blockSize = 128;
    int gridSize  = (N + blockSize - 1) / blockSize;
    kernelExample<<<gridSize, blockSize>>>(d_data, N);
    cudaDeviceSynchronize(); // 等待 kernel 完成

    // 4. （可选）检验结果
    //    先在 CPU 上分配一份 host 数组来接收结果
    float *h_p = new float[N];
    cudaMemcpy(h_p, d_data.p, N * sizeof(float), cudaMemcpyDeviceToHost);
    // 在此可打印部分结果看是否符合预期
    std::cout << "p[0] = " << h_p[0] << std::endl;

    // 5. 释放内存
    delete[] h_p; // 释放 CPU 上的数组
    freeDeviceMemory(d_data); // 释放 GPU 上的数组

    return 0;
}
