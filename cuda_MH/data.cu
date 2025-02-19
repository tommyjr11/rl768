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
    if (maxSpeed < 1e-15f) {
        return 1.0e10f; // 给一个很大的dt
    }
    // 选一个最小网格尺度
    float minDxDy = fminf(dx, dy);
    float dt = C * minDxDy / maxSpeed;
    return dt;
}

// 内核函数：更新左右边界
__global__ void boundary_left_right(solVectors u, int truenx, int trueny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < trueny) {
        int rowStart = i * truenx;
        // 左边界：将第0列和第1列赋值为第2列的值
        u.p[rowStart + 0] = u.p[rowStart + 2];
        u.p[rowStart + 1] = u.p[rowStart + 2];
        u.rho[rowStart + 0] = u.rho[rowStart + 2];
        u.rho[rowStart + 1] = u.rho[rowStart + 2];
        u.vx[rowStart + 0] = u.vx[rowStart + 2];
        u.vx[rowStart + 1] = u.vx[rowStart + 2];
        u.vy[rowStart + 0] = u.vy[rowStart + 2];
        u.vy[rowStart + 1] = u.vy[rowStart + 2];
        
        // 右边界：将倒数第1列和倒数第2列赋值为倒数第3列的值
        u.p[rowStart + (truenx - 2)] = u.p[rowStart + (truenx - 3)];
        u.p[rowStart + (trueny - 1)] = u.p[rowStart + (trueny - 3)];
        u.rho[rowStart + (truenx - 2)] = u.rho[rowStart + (truenx - 3)];
        u.rho[rowStart + (trueny - 1)] = u.rho[rowStart + (trueny - 3)];
        u.vx[rowStart + (truenx - 2)] = u.vx[rowStart + (truenx - 3)];
        u.vx[rowStart + (trueny - 1)] = u.vx[rowStart + (trueny - 3)];
        u.vy[rowStart + (truenx - 2)] = u.vy[rowStart + (truenx - 3)];
        u.vy[rowStart + (trueny - 1)] = u.vy[rowStart + (trueny - 3)];
    }
}

// 内核函数：更新上下边界
__global__ void boundary_top_bottom(solVectors u, int truenx, int trueny) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < truenx) {
        // 上边界：将第0行和第1行赋值为第2行的值
        u.p[0 * truenx + j] = u.p[2 * truenx + j];
        u.p[1 * truenx + j] = u.p[2 * truenx + j];
        u.rho[0 * truenx + j] = u.rho[2 * truenx + j];
        u.rho[1 * truenx + j] = u.rho[2 * truenx + j];
        u.vx[0 * truenx + j] = u.vx[2 * truenx + j];
        u.vx[1 * truenx + j] = u.vx[2 * truenx + j];
        u.vy[0 * truenx + j] = u.vy[2 * truenx + j];
        u.vy[1 * truenx + j] = u.vy[2 * truenx + j];
        // 下边界：将倒数第1行和倒数第2行赋值为倒数第3行的值
        u.p[(trueny - 2) * truenx + j] = u.p[(trueny - 3) * truenx + j];
        u.p[(trueny - 1) * truenx + j] = u.p[(trueny - 3) * truenx + j];
        u.rho[(trueny - 2) * truenx + j] = u.rho[(trueny - 3) * truenx + j];
        u.rho[(trueny - 1) * truenx + j] = u.rho[(trueny - 3) * truenx + j];
        u.vx[(trueny - 2) * truenx + j] = u.vx[(trueny - 3) * truenx + j];
        u.vx[(trueny - 1) * truenx + j] = u.vx[(trueny - 3) * truenx + j];
    }
}

// 边界条件更新函数：接收指向GPU内存的指针
void applyBoundaryConditions(solVectors &d_u) {
    int threadsPerBlock = 128;
    int truenx = nx + 4;
    int trueny = ny + 4;
    // 更新左右边界：每个线程处理一行
    int blocksLR = ((ny+4) + threadsPerBlock - 1) / threadsPerBlock;
    boundary_left_right<<<blocksLR, threadsPerBlock>>>(d_u, truenx, trueny);
    // 更新上下边界：每个线程处理一列
    int blocksTB = (nx + threadsPerBlock - 1) / threadsPerBlock;
    boundary_top_bottom<<<blocksTB, threadsPerBlock>>>(d_u, truenx, trueny);
    // 等待内核执行完成
    cudaDeviceSynchronize();
}