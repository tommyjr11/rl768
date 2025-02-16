#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_functions.h"

__global__ void kernel(float *d_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_data[idx] *= 2.0f;  // 示例：把数据乘2
    }
}

void my_cuda_function(float *h_data, int size) {
    float *d_data;
    cudaMalloc((void**)&d_data, size * sizeof(float));
    cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    kernel<<<blocks, threads>>>(d_data, size);

    cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}
