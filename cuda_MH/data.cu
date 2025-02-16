#include "data.h"
#include "constants.h"  
#include <iostream>
#include <vector>
#include <cmath>       



void allocateDeviceMemory(solVectors &d_data) {
    CUDA_CHECK(cudaMalloc((void**)&(d_data.rho), (nx+4) * (ny+4) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&(d_data.vx),  (nx+4) * (ny+4) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&(d_data.vy),  (nx+4) * (ny+4) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&(d_data.p),   (nx+4) * (ny+4) * sizeof(float)));
}

void freeDeviceMemory(solVectors &d_data) {
    CUDA_CHECK(cudaFree(d_data.rho));
    CUDA_CHECK(cudaFree(d_data.vx));
    CUDA_CHECK(cudaFree(d_data.vy));
    CUDA_CHECK(cudaFree(d_data.p));
}

void initDataAndCopyToGPU(solVectors &d_data)
{
    std::vector<float> h_rho((nx+4) * (ny+4), 0.0f);
    std::vector<float> h_vx ((nx+4) * (ny+4), 0.0f);
    std::vector<float> h_vy ((nx+4) * (ny+4), 0.0f);
    std::vector<float> h_p  ((nx+4) * (ny+4), 0.0f);

    // 初始化
    for (int j = 0; j < ny+4; j++) {
        for (int i = 0; i < nx+4; i++) {
            int idx = j * (nx+4) + i;

            // 将(i,j)映射到物理坐标 (x, y)
            float x = (i - ghost + 0.5f) * dx; 
            float y = (j - ghost + 0.5f) * dy;

            // 根据坐标区域，给出不同初值（示例）
            if (x < 0.5f) {
                if (y < 0.5f) {
                    h_rho[idx] = 0.138f;
                    h_vx [idx] = 1.206f;
                    h_vy [idx] = 1.206f;
                    h_p  [idx] = 0.029f;
                } else {
                    h_rho[idx] = 0.5323f;
                    h_vx [idx] = 1.206f;
                    h_vy [idx] = 0.0f;
                    h_p  [idx] = 0.3f;
                }
            } else {
                if (y < 0.5f) {
                    h_rho[idx] = 0.5323f;
                    h_vx [idx] = 0.0f;
                    h_vy [idx] = 1.206f;
                    h_p  [idx] = 0.3f;
                } else {
                    h_rho[idx] = 1.5f;
                    h_vx [idx] = 0.0f;
                    h_vy [idx] = 0.0f;
                    h_p  [idx] = 1.5f;
                }
            }
        }
    }

    // 拷贝到 GPU
    size_t sizeBytes = (nx+4) * (ny+4) * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_data.rho, h_rho.data(), sizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data.vx,  h_vx.data(),  sizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data.vy,  h_vy.data(),  sizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data.p,   h_p.data(),   sizeBytes, cudaMemcpyHostToDevice));
}

__global__ void getMaxSpeedKernel(
    const float* __restrict__ rho,
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const float* __restrict__ p,
    float* __restrict__ blockMax,
    int totalSize,
    float r)
{
    extern __shared__ float sdata[];

    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    int tid = threadIdx.x;

    float localMax = 0.0f;

    // **优化1: grid-stride 循环**
    for (int idx = globalThreadId; idx < totalSize; idx += step) {
        float c   = sqrtf(r * __ldg(&p[idx]) / __ldg(&rho[idx]));  // **优化2: __ldg() 提高访存效率**
        float spx = fabsf(__ldg(&vx[idx])) + c;
        float spy = fabsf(__ldg(&vy[idx])) + c;
        localMax  = fmaxf(localMax, fmaxf(spx, spy));
    }

    // **优化3: Warp-level reduction**
    sdata[tid] = localMax;
    __syncthreads();

    // **Warp 级别归约**
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        localMax = fmaxf(localMax, __shfl_down_sync(0xffffffff, localMax, offset));
    }

    // **优化4: 使用 warp shuffle 归约到 warp 0**
    if ((tid % warpSize) == 0) {
        sdata[tid / warpSize] = localMax;
    }
    __syncthreads();

    // **仅 block 内 thread 0 进行最终归约**
    if (tid == 0) {
        for (int i = 1; i < blockDim.x / warpSize; i++) {
            localMax = fmaxf(localMax, sdata[i]);
        }
        blockMax[blockIdx.x] = localMax;
    }
}


float getmaxspeedGPU(const solVectors &d_data, float r)
{
    // 这里简化一下，直接把 totalSize = (nx+4)*(ny+4)
    int totalSize = (nx+4) * (ny+4);

    int blockSize = 64;
    int gridSize  = (totalSize + blockSize - 1) / blockSize;

    float *d_blockMax = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockMax, gridSize * sizeof(float)));

    int sharedMemSize = blockSize * sizeof(float);
    
    getMaxSpeedKernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_data.rho,
        d_data.vx,
        d_data.vy,
        d_data.p,
        d_blockMax,
        totalSize,
        r
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
    }
    std::vector<float> h_blockMax(gridSize, 0.0f);
    CUDA_CHECK(cudaMemcpy(h_blockMax.data(), d_blockMax, gridSize * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_blockMax));

    float maxSpeed = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        maxSpeed = fmaxf(maxSpeed, h_blockMax[i]);
    }

    return maxSpeed;
}

float getdtGPU(const solVectors &d_data, float r)
{
    float maxSpeed = getmaxspeedGPU(d_data, r);
    std::cout<<"maxSpeed: "<<maxSpeed<<std::endl;

    // 避免除以0
    if (maxSpeed < 1e-15f) {
        return 1.0e10f; // 给一个很大的dt
    }

    // 选一个最小网格尺度
    float minDxDy = fminf(dx, dy);
    float dt = C * minDxDy / maxSpeed;
    return dt;
}