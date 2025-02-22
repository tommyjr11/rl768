#include "data.h"
#include "constants.h"  
#include <iostream>
#include <vector>
#include <cmath>       



void allocateDeviceMemory(solVectors &d_data_pri, solVectors &d_data_con) {
    CUDA_CHECK(cudaMalloc((void**)&(d_data_pri.rho), (nx+4) * (ny+4) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&(d_data_pri.vx),  (nx+4) * (ny+4) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&(d_data_pri.vy),  (nx+4) * (ny+4) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&(d_data_pri.p),   (nx+4) * (ny+4) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&(d_data_con.rho), (nx+4) * (ny+4) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&(d_data_con.vx),  (nx+4) * (ny+4) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&(d_data_con.vy),  (nx+4) * (ny+4) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&(d_data_con.p),   (nx+4) * (ny+4) * sizeof(float)));
}

void freeDeviceMemory(solVectors &d_data_pri, solVectors &d_data_con) {
    CUDA_CHECK(cudaFree(d_data_pri.rho));
    CUDA_CHECK(cudaFree(d_data_pri.vx));
    CUDA_CHECK(cudaFree(d_data_pri.vy));
    CUDA_CHECK(cudaFree(d_data_pri.p));
    CUDA_CHECK(cudaFree(d_data_con.rho));
    CUDA_CHECK(cudaFree(d_data_con.vx));
    CUDA_CHECK(cudaFree(d_data_con.vy));
    CUDA_CHECK(cudaFree(d_data_con.p));
}

__device__ void get_con(const float *pri, float *con)
{
    con[0] = pri[0];
    con[1] = pri[0]*pri[1];
    con[2] = pri[0]*pri[2];
    con[3] = 0.5*pri[0]*(pow(pri[1],2)+pow(pri[2],2))+pri[3]/(1.4-1);
}

__device__ void get_pri(const float *con, float *pri)
{
    pri[0] = con[0];
    pri[1] = con[1]/con[0];
    pri[2] = con[2]/con[0];
    pri[3] = (1.4-1)*(con[3]-0.5*con[0]*(pow(pri[1],2)+pow(pri[2],2)));
}

__global__ void kernel_pri2con(const solVectors d_data_pri, solVectors d_data_con, 
    int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx+4 && j < ny+4) {
    int idx = j * (nx+4) + i;
    float pri[4];
    pri[0] = d_data_pri.rho[idx];  // rho
    pri[1] = d_data_pri.vx [idx];  // vx
    pri[2] = d_data_pri.vy [idx];  // vy
    pri[3] = d_data_pri.p  [idx];  // p
    float con[4];
    get_con(pri,con);
    d_data_con.rho[idx] = con[0];  // rho
    d_data_con.vx [idx] = con[1];  // rho*vx
    d_data_con.vy [idx] = con[2];  // rho*vy
    d_data_con.p  [idx] = con[3];  // E (总能量)
    }
}

__global__ void kernel_con2pri(const solVectors d_data_con, solVectors d_data_pri, 
    int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx+4 && j < ny+4) {
    int idx = j * (nx+4) + i;
    float con[4];
    con[0] = d_data_con.rho[idx];  // rho
    con[1] = d_data_con.vx [idx];  // rho*vx
    con[2] = d_data_con.vy [idx];  // rho*vy
    con[3] = d_data_con.p  [idx];  // E (总能量)
    float pri[4];
    get_pri(con,pri);
    d_data_pri.rho[idx] = pri[0];  // rho
    d_data_pri.vx [idx] = pri[1];  // vx
    d_data_pri.vy [idx] = pri[2];  // vy
    d_data_pri.p  [idx] = pri[3];  // p
    }
}

void initDataAndCopyToGPU(solVectors &d_data_pri,solVectors d_data_con)
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
    CUDA_CHECK(cudaMemcpy(d_data_pri.rho, h_rho.data(), sizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data_pri.vx,  h_vx.data(),  sizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data_pri.vy,  h_vy.data(),  sizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data_pri.p,   h_p.data(),   sizeBytes, cudaMemcpyHostToDevice));
    // 将 d_data_pri 的数据拷贝到 d_data_con
    dim3 blockSize(16, 16);
    dim3 gridSize((nx+4+15)/16, (ny+4+15)/16);
    kernel_pri2con<<<gridSize, blockSize>>>(d_data_pri, d_data_con, nx, ny);
    cudaDeviceSynchronize();
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


float getmaxspeedGPU(const solVectors &d_data_pri, float r)
{
    // 这里简化一下，直接把 totalSize = (nx+4)*(ny+4)
    int totalSize = (nx+4) * (ny+4);

    int blockSize = 64;
    int gridSize  = (totalSize + blockSize - 1) / blockSize;

    float *d_blockMax = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockMax, gridSize * sizeof(float)));

    int sharedMemSize = blockSize * sizeof(float);
    
    getMaxSpeedKernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_data_pri.rho,
        d_data_pri.vx,
        d_data_pri.vy,
        d_data_pri.p,
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

float getdtGPU(const solVectors &d_data_pri, float r)
{
    float maxSpeed = getmaxspeedGPU(d_data_pri, r);
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



__device__ float limiterL2(float smaller, float larger) {
    if (larger == 0.0)
        return (smaller == 0.0) ? 0.0 : 1.0;
    float R = smaller / larger;
    return fminf(fmaxf(R, 0.0), 1.0);
}

__device__ float limiterR2(float smaller, float larger) {
    if (larger == 0.0)
        return 0.0;
        float R = smaller / larger;
    return (R <= 0.0) ? 0.0 : ((R <= 1.0) ? R : fmin(1.0, 2.0/(1.0+R)));
}

__device__ void get_flux_x(const float *pri, float *flux) {
    flux[0] = pri[0]*pri[1];
    flux[1] = pri[0]*pri[1]*pri[1] + pri[3];
    flux[2] = pri[0]*pri[1]*pri[2];
    float Energy = 0.5*pri[0]*(pri[1]*pri[1] + pri[2]*pri[2]) + pri[3]/(1.4-1.0);
    flux[3] = pri[1]*(pri[3] + Energy);
}

__device__ void get_flux_y(const float *pri, float *flux) {
    flux[0] = pri[0]*pri[2];
    flux[1] = pri[0]*pri[1]*pri[2];
    flux[2] = pri[0]*pri[2]*pri[2] + pri[3];
    float Energy = 0.5*pri[0]*(pri[1]*pri[1] + pri[2]*pri[2]) + pri[3]/(1.4-1.0);
    flux[3] = pri[2]*(pri[3] + Energy);
}

__global__ void computeHalftimeKernel_x(
    const solVectors &d_data_con,
    solVectors d_half_uL,
    solVectors d_half_uR, 
    float dt,
    float dx,
    int nx, int ny
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= (nx+3) || j >= (ny+4)) {
        return;
    }
    int stride = (nx + 4);
    int idx      = j*stride + i;
    int idx_left = j*stride + (i - 1);
    int idx_right= j*stride + (i + 1);

    // --- Step 1: 读取 con(i,j), con(i-1,j), con(i+1,j) ---
    float conM[4];  // con(i,j)
    float conL[4];  // con(i-1,j)
    float conR[4];  // con(i+1,j)
    conM[0] = d_data_con.rho[idx];
    conM[1] = d_data_con.vx [idx];  // 这里 vx 里实际存的是 rho*u
    conM[2] = d_data_con.vy [idx];  // 这里 vy 里实际存的是 rho*v
    conM[3] = d_data_con.p  [idx];  // E (总能量)

    conL[0] = d_data_con.rho[idx_left];
    conL[1] = d_data_con.vx [idx_left];
    conL[2] = d_data_con.vy [idx_left];
    conL[3] = d_data_con.p  [idx_left];

    conR[0] = d_data_con.rho[idx_right];
    conR[1] = d_data_con.vx [idx_right];
    conR[2] = d_data_con.vy [idx_right];
    conR[3] = d_data_con.p  [idx_right];

    // --- Step 2: 斜率限制，得到 tempL, tempR (仍在保守量空间) ---
    float tempL[4], tempR[4];
    for (int k = 0; k < 4; k++) {
        float temp1 = conM[k] - conL[k];  // i - (i-1)
        float temp2 = conR[k] - conM[k];  // (i+1) - i
        float di = 0.5f * (temp1 + temp2);

        // 这里分别调用 limiterL2 / limiterR2：
        float phiL = limiterL2(temp1, temp2);
        float phiR = limiterR2(temp1, temp2);

        // 得到左右临时状态
        tempL[k] = conM[k] - 0.5f * di * phiL;
        tempR[k] = conM[k] + 0.5f * di * phiR;
    }
    // --- Step 3: 将 tempL, tempR 转为原始量 priL, priR，并计算通量 fluxL, fluxR ---
    float priL[4], priR[4];
    get_pri(tempL, priL);
    get_pri(tempR, priR);

    float fluxL[4], fluxR[4];
    get_flux_x(priL, fluxL);
    get_flux_x(priR, fluxR);

    // --- Step 4: 半步更新 (回到保守量空间) ---
    // tempL, tempR 各减去 0.5*(dt/dx)*(fluxR - fluxL)
    for (int k = 0; k < 4; k++) {
        float delta = 0.5f * (dt / dx) * (fluxR[k] - fluxL[k]);
        tempL[k] = tempL[k] - delta;
        tempR[k] = tempR[k] - delta;
    }

    // --- Step 5: 把结果存到 half_uL, half_uR 里 ---
    // half_uL, half_uR 大小可能是 (nx-2)*ny
    // 所以对应的一维索引 out_idx = out_j*(nx-2) + out_i
    // 这里 out_j = j (0 <= j < ny)
    if (j >= 0 && (i-1) < (nx - 2)) {
        int out_idx = j*(nx - 2) + (i-1);
        // 写入 half_uL
        d_half_uL.rho[out_idx] = tempL[0];
        d_half_uL.vx [out_idx] = tempL[1];
        d_half_uL.vy [out_idx] = tempL[2];
        d_half_uL.p  [out_idx] = tempL[3];

        // 写入 half_uR
        d_half_uR.rho[out_idx] = tempR[0];
        d_half_uR.vx [out_idx] = tempR[1];
        d_half_uR.vy [out_idx] = tempR[2];
        d_half_uR.p  [out_idx] = tempR[3];
    }
}

void computeHalftime(
    const solVectors &d_data_con,
    solVectors &d_half_uL,
    solVectors &d_half_uR,
    float dt,
    int choice
)
{
    if (choice == 1)
    {
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uL.rho), (nx+2) * (ny+4) * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uL.vx),  (nx+2) * (ny+4) * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uL.vy),  (nx+2) * (ny+4) * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uL.p),   (nx+2) * (ny+4) * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uR.rho), (nx+2) * (ny+4) * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uR.vx),  (nx+2) * (ny+4) * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uR.vy),  (nx+2) * (ny+4) * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uR.p),   (nx+2) * (ny+4) * sizeof(float)));
        dim3 block(16, 16);
        dim3 grid( (nx+block.x-1)/block.x, (ny+block.y-1)/block.y );
        computeHalftimeKernel_x<<<grid, block>>>(
            d_data_con,    
            d_half_uL,     
            d_half_uR,     
            dt, dx, 
            nx, ny
        );
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
            exit(-1);
        }
    }
}

