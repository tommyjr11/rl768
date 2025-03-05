#include "data.h"
#include <iostream>
#include <vector>
#include <cmath>       
#include <filesystem>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>


void allocateDeviceMemory(solVectors &d_data_pri, solVectors &d_data_con) {
    CUDA_CHECK(cudaMalloc((void**)&(d_data_pri.rho), (nx+4) * (ny+4) * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&(d_data_pri.vx),  (nx+4) * (ny+4) * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&(d_data_pri.vy),  (nx+4) * (ny+4) * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&(d_data_pri.p),   (nx+4) * (ny+4) * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&(d_data_con.rho), (nx+4) * (ny+4) * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&(d_data_con.vx),  (nx+4) * (ny+4) * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&(d_data_con.vy),  (nx+4) * (ny+4) * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&(d_data_con.p),   (nx+4) * (ny+4) * sizeof(double)));
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

__device__ void get_con(const double *pri, double *con)
{
    con[0] = pri[0];
    con[1] = pri[0]*pri[1];
    con[2] = pri[0]*pri[2];
    con[3] = 0.5*pri[0]*(pow(pri[1],2)+pow(pri[2],2))+pri[3]/(1.4-1);
}

__device__ void get_pri(const double *con, double *pri)
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
    double pri[4];
    pri[0] = d_data_pri.rho[idx];  // rho
    pri[1] = d_data_pri.vx [idx];  // vx
    pri[2] = d_data_pri.vy [idx];  // vy
    pri[3] = d_data_pri.p  [idx];  // p
    double con[4];
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
    double con[4];
    con[0] = d_data_con.rho[idx];  // rho
    con[1] = d_data_con.vx [idx];  // rho*vx
    con[2] = d_data_con.vy [idx];  // rho*vy
    con[3] = d_data_con.p  [idx];  // E (总能量)
    double pri[4];
    get_pri(con,pri);
    d_data_pri.rho[idx] = pri[0];  // rho
    d_data_pri.vx [idx] = pri[1];  // vx
    d_data_pri.vy [idx] = pri[2];  // vy
    d_data_pri.p  [idx] = pri[3];  // p
    }
}

// void initDataAndCopyToGPU1(solVectors &d_data_pri,solVectors d_data_con)
// {
//     std::vector<double> h_rho((nx+4) * (ny+4), 0.0);
//     std::vector<double> h_vx ((nx+4) * (ny+4), 0.0);
//     std::vector<double> h_vy ((nx+4) * (ny+4), 0.0);
//     std::vector<double> h_p  ((nx+4) * (ny+4), 0.0);

//     // 初始化
//     for (int j = 0; j < ny+4; j++) {
//         for (int i = 0; i < nx+4; i++) {
//             int idx = j * (nx+4) + i;

//             // 将(i,j)映射到物理坐标 (x, y)
//             double x = (i - ghost + 0.5) * dx; 
//             double y = (j - ghost + 0.5) * dy;

//             // 根据坐标区域，给出不同初值（示例）
//             if (x < 0.5) {
//                 if (y < 0.5) {
//                     h_rho[idx] = 0.138;
//                     h_vx [idx] = 1.206;
//                     h_vy [idx] = 1.206;
//                     h_p  [idx] = 0.029;
//                 } else {
//                     h_rho[idx] = 0.5323;
//                     h_vx [idx] = 1.206;
//                     h_vy [idx] = 0.0;
//                     h_p  [idx] = 0.3;
//                 }
//             } else {
//                 if (y < 0.5) {
//                     h_rho[idx] = 0.5323;
//                     h_vx [idx] = 0.0;
//                     h_vy [idx] = 1.206;
//                     h_p  [idx] = 0.3;
//                 } else {
//                     h_rho[idx] = 1.5;
//                     h_vx [idx] = 0.0;
//                     h_vy [idx] = 0.0;
//                     h_p  [idx] = 1.5;
//                 }
//             }
//         }
//     }

//     // 拷贝到 GPU
//     size_t sizeBytes = (nx+4) * (ny+4) * sizeof(double);
//     CUDA_CHECK(cudaMemcpy(d_data_pri.rho, h_rho.data(), sizeBytes, cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_data_pri.vx,  h_vx.data(),  sizeBytes, cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_data_pri.vy,  h_vy.data(),  sizeBytes, cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_data_pri.p,   h_p.data(),   sizeBytes, cudaMemcpyHostToDevice));
//     // 将 d_data_pri 的数据拷贝到 d_data_con
//     dim3 blockSize(16, 16);
//     dim3 gridSize((nx+4+15)/16, (ny+4+15)/16);
//     kernel_pri2con<<<gridSize, blockSize>>>(d_data_pri, d_data_con, nx, ny);
//     cudaDeviceSynchronize();
// }
void initDataAndCopyToGPU2(solVectors &d_data_pri, solVectors d_data_con)
{
    std::vector<double> h_rho((nx+4) * (ny+4), 0.0);
    std::vector<double> h_vx ((nx+4) * (ny+4), 0.0);
    std::vector<double> h_vy ((nx+4) * (ny+4), 0.0);
    std::vector<double> h_p  ((nx+4) * (ny+4), 0.0);

    // 初始化
    for (int j = 0; j < ny+4; j++) {
        for (int i = 0; i < nx+4; i++) {
            int idx = j * (nx+4) + i;

            // 将(i,j)映射到物理坐标 (x, y)，中心点略加0.5 * dx(or dy)
            double x = (i - ghost + 0.5f) * dx; 
            double y = (j - ghost + 0.5f) * dy;

            // 先设为空气
            if (x < xShock) {
                // 激波后空气(左侧)
                h_rho[idx] = rhoPost;
                h_vx [idx] = uPost;
                h_vy [idx] = vPost;
                h_p  [idx] = pPost;
            } else {
                // 未受激波空气(右侧)
                h_rho[idx] = rhoAir;
                h_vx [idx] = uAir;
                h_vy [idx] = vAir;
                h_p  [idx] = pAir;
            }

            // 判断是否在气泡内（这一步覆盖掉空气的设定）
            double dxBubble = x - bubbleXc;
            double dyBubble = y - bubbleYc;
            if ( (dxBubble*dxBubble + dyBubble*dyBubble) <= bubbleR*bubbleR ) {
                // 落在气泡区域
                h_rho[idx] = rhoHe; 
                // 氦气泡与外界等压、速度为零
                h_vx [idx] = 0.0;
                h_vy [idx] = 0.0;
                h_p  [idx] = pAir;  // 与外界相同压强
            }
        }
    }

    // 拷贝到 GPU
    size_t sizeBytes = (nx+4) * (ny+4) * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_data_pri.rho, h_rho.data(), sizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data_pri.vx,  h_vx.data(),  sizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data_pri.vy,  h_vy.data(),  sizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data_pri.p,   h_p.data(),   sizeBytes, cudaMemcpyHostToDevice));

    // 将 d_data_pri 的数据转换为守恒量并拷到 d_data_con
    dim3 blockSize(16, 16);
    dim3 gridSize((nx+4+15)/16, (ny+4+15)/16);
    kernel_pri2con<<<gridSize, blockSize>>>(d_data_pri, d_data_con, nx, ny);
    cudaDeviceSynchronize();
}
__global__ void getMaxSpeedKernel(
    const double* __restrict__ rho,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    const double* __restrict__ p,
    double* __restrict__ blockMax,
    int totalSize,
    double r)
{
    extern __shared__ double sdata[];

    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    int tid = threadIdx.x;

    double localMax = 0.0;

    // **优化1: grid-stride 循环**
    for (int idx = globalThreadId; idx < totalSize; idx += step) {
        double c   = sqrt(r * __ldg(&p[idx]) / __ldg(&rho[idx]));  // **优化2: __ldg() 提高访存效率**
        double spx = fabs(__ldg(&vx[idx])) + c;
        double spy = fabs(__ldg(&vy[idx])) + c;
        localMax  = fmax(localMax, fmax(spx, spy));
    }

    // **优化3: Warp-level reduction**
    sdata[tid] = localMax;
    __syncthreads();

    // **Warp 级别归约**
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        localMax = fmax(localMax, __shfl_down_sync(0xffffffff, localMax, offset));
    }

    // **优化4: 使用 warp shuffle 归约到 warp 0**
    if ((tid % warpSize) == 0) {
        sdata[tid / warpSize] = localMax;
    }
    __syncthreads();

    // **仅 block 内 thread 0 进行最终归约**
    if (tid == 0) {
        for (int i = 1; i < blockDim.x / warpSize; i++) {
            localMax = fmax(localMax, sdata[i]);
        }
        blockMax[blockIdx.x] = localMax;
    }
}





double getmaxspeedGPU(const solVectors &d_data_pri, double r)
{
    int totalSize = (nx+4) * (ny+4);
    int blockSize = 64;
    int gridSize  = (totalSize + blockSize - 1) / blockSize;
    double *d_blockMax = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockMax, gridSize * sizeof(double)));
    int sharedMemSize = blockSize * sizeof(double);
    getMaxSpeedKernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_data_pri.rho,
        d_data_pri.vx,
        d_data_pri.vy,
        d_data_pri.p,
        d_blockMax,
        totalSize,
        r
    );
    // CUDA_CHECK(cudaDeviceSynchronize());
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
    // }
    std::vector<double> h_blockMax(gridSize, 0.0);
    CUDA_CHECK(cudaMemcpy(h_blockMax.data(), d_blockMax, gridSize * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_blockMax));

    double maxSpeed = 0.0;
    for (int i = 0; i < gridSize; i++) {
        maxSpeed = fmax(maxSpeed, h_blockMax[i]);
    }

    return maxSpeed;
}

double getdtGPU(const solVectors &d_data_pri, double r)
{
    double maxSpeed = getmaxspeedGPU(d_data_pri, r);
    if (maxSpeed < 1e-15) {
        return 1.0e10; // 给一个很大的dt
    }
    // 选一个最小网格尺度
    double minDxDy = fmin(dx, dy);
    double dt = C * minDxDy / maxSpeed;
    return dt;
}

// 内核函数：更新左右边界
__global__ void boundary_left_right(solVectors u, int truenx, int trueny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < trueny)
    {
        int rowStart = i * truenx;

        // --- 左边界 ---
        // col 1 <- col 2
        u.p  [rowStart + 1] = u.p  [rowStart + 2];
        u.rho[rowStart + 1] = u.rho[rowStart + 2];
        u.vx [rowStart + 1] = u.vx [rowStart + 2];
        u.vy [rowStart + 1] = u.vy [rowStart + 2];

        // col 0 <- col 1  (此时 col 1 已经更新)
        u.p  [rowStart + 0] = u.p  [rowStart + 3];
        u.rho[rowStart + 0] = u.rho[rowStart + 3];
        u.vx [rowStart + 0] = u.vx [rowStart + 3];
        u.vy [rowStart + 0] = u.vy [rowStart + 3];

        // --- 右边界 ---
        // col (truenx - 2) <- col (truenx - 3)
        u.p  [rowStart + (truenx - 2)] = u.p  [rowStart + (truenx - 3)];
        u.rho[rowStart + (truenx - 2)] = u.rho[rowStart + (truenx - 3)];
        u.vx [rowStart + (truenx - 2)] = u.vx [rowStart + (truenx - 3)];
        u.vy [rowStart + (truenx - 2)] = u.vy [rowStart + (truenx - 3)];

        // col (truenx - 1) <- col (truenx - 2)
        u.p  [rowStart + (truenx - 1)] = u.p  [rowStart + (truenx - 4)];
        u.rho[rowStart + (truenx - 1)] = u.rho[rowStart + (truenx - 4)];
        u.vx [rowStart + (truenx - 1)] = u.vx [rowStart + (truenx - 4)];
        u.vy [rowStart + (truenx - 1)] = u.vy [rowStart + (truenx - 4)];
    }
}

// 内核函数：更新上下边界
__global__ void boundary_top_bottom(solVectors u, int truenx, int trueny)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < truenx)
    {
        // --- 上边界 ---
        // row 1 <- row 2
        u.p  [1 * truenx + j] = u.p  [2 * truenx + j];
        u.rho[1 * truenx + j] = u.rho[2 * truenx + j];
        u.vx [1 * truenx + j] = u.vx [2 * truenx + j];
        u.vy [1 * truenx + j] = u.vy [2 * truenx + j];

        // row 0 <- row 1  (此时 row 1 已经更新)
        u.p  [0 * truenx + j] = u.p  [3 * truenx + j];
        u.rho[0 * truenx + j] = u.rho[3 * truenx + j];
        u.vx [0 * truenx + j] = u.vx [3 * truenx + j];
        u.vy [0 * truenx + j] = u.vy [3 * truenx + j];

        // --- 下边界 ---
        // row (trueny - 2) <- row (trueny - 3)
        u.p  [(trueny - 2) * truenx + j] = u.p  [(trueny - 3) * truenx + j];
        u.rho[(trueny - 2) * truenx + j] = u.rho[(trueny - 3) * truenx + j];
        u.vx [(trueny - 2) * truenx + j] = u.vx [(trueny - 3) * truenx + j];
        u.vy [(trueny - 2) * truenx + j] = u.vy [(trueny - 3) * truenx + j];

        // row (trueny - 1) <- row (trueny - 2)
        u.p  [(trueny - 1) * truenx + j] = u.p  [(trueny - 4) * truenx + j];
        u.rho[(trueny - 1) * truenx + j] = u.rho[(trueny - 4) * truenx + j];
        u.vx [(trueny - 1) * truenx + j] = u.vx [(trueny - 4) * truenx + j];
        u.vy [(trueny - 1) * truenx + j] = u.vy [(trueny - 4) * truenx + j];
    }
}

// 边界条件更新函数
void applyBoundaryConditions(solVectors &d_u)
{
    // nx、ny 是实际物理网格的大小
    // 如果您在分配时加了 4 个ghost cell，则
    int truenx = nx + 4;
    int trueny = ny + 4;

    int threadsPerBlock = 64;

    // 更新左右边界：每个线程处理一行
    int blocksLR = (trueny + threadsPerBlock - 1) / threadsPerBlock;
    boundary_left_right<<<blocksLR, threadsPerBlock>>>(d_u, truenx, trueny);

    // 更新上下边界：每个线程处理一列
    int blocksTB = (truenx + threadsPerBlock - 1) / threadsPerBlock;
    boundary_top_bottom<<<blocksTB, threadsPerBlock>>>(d_u, truenx, trueny);

    // 等待内核执行完成并检查错误
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed at boundary: "
                  << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

// __device__ double limiterL2(double smaller, double larger) {
//     // 把 "larger == 0" 的情况合并成一次三元运算
//     // 若 (larger == 0 && smaller != 0) => slope = 1
//     // 若 (larger == 0 && smaller == 0) => slope = 0
//     // 否则 => slope = smaller / larger
//     double slope = (larger == 0.0) 
//                      ? ((smaller == 0.0) ? 0.0 : 1.0)
//                      : (smaller / larger);

//     // 把负值直接截断为 0
//     slope = fmax(0.0, slope);

//     // 当 slope > 1 的时候，用 2*slope/(1+slope)，否则就用 slope，
//     // 再额外与 1.0 做一次 fmin。
//     // 这样就相当于把 “0 < R <= 1 => R；R > 1 => 2R/(1+R) 再与1比较” 的逻辑合并了
//     double limiter = 2.0 * slope / (1.0 + slope);          // 对应 "2R/(1+R)"
//     double limited = fmin(slope, limiter);                // 相当于 if (slope > 1) 走 limiter，否则走 slope
//     return fmin(1.0, limited);        
// }

// __device__ double limiterR2(double smaller, double larger) {
//     // 若 larger == 0.0，无论 smaller 是否为0，都令 slope=0，效果等同：
//     //   smaller == 0 => slope=0
//     //   smaller != 0 => return 0（一样是0）
//     double slope = (larger == 0.0) ? 0.0 : (smaller / larger);

//     // 负值截断为0
//     slope = fmax(0.0, slope);

//     // 若 slope > 1 => 用 2/(1+slope)，否则是 slope，
//     // 最后再与 1.0 做比较
//     double limiter = 2.0 / (1.0 + slope);      
//     double limited = fmin(slope, limiter);
//     return fmin(1.0, limited);
// }

__device__ double limiterL2(double smaller, double larger) {
    double R_slope = 0.0;
    if (smaller == 0 && larger == 0){
        R_slope = 0.0;
    }
    else if (larger == 0 && smaller != 0){
        return 0.0;
    }
    else{
        R_slope = smaller/larger;
    }
    if (R_slope <= 0){
        return 0.0;
    }
    else if (R_slope <= 1){
        return R_slope;
    }
    else {
        double temp2 = temp2 = 2/(1+R_slope);
        return min(1.0, temp2);
        }  
}

__device__ double limiterR2(double smaller, double larger) {
    double R_slope = 0.0;
    if (smaller == 0 && larger == 0){
        R_slope = 0.0;
    }
    else if (larger == 0 && smaller != 0){
        return 0.0;
    }
    else{
        R_slope = smaller/larger;
        }
    if (R_slope <= 0){
        return 0.0;
    }
    else if (R_slope <= 1)
    {
        return R_slope;
    }
    else 
    {
        double temp2 = 2/(1+R_slope);
        return min(1.0, temp2);
    }
}

// __device__ double limiterL2(double smaller, double larger)
// {
//     // 1) 若 larger == 0.0，则根据 smaller 是否为 0 决定返回值
//     //    原逻辑： (larger==0 && smaller==0) => 0, (larger==0 && smaller!=0) => 1
//     if (larger == 0.0) {
//         if (smaller == 0.0) {
//             return 0.0;
//         } else {
//             return 1.0;
//         }
//     }

//     // 2) 计算 slope
//     double slope = smaller / larger;

//     // 3) 若 slope <= 0，直接返回 0
//     if (slope <= 0.0) {
//         return 0.0;
//     }

//     // 4) 使用“少分支”公式
//     //    若 slope <= 1 => 返回 slope
//     //    若 slope > 1  => 返回 2*slope / (1 + slope)，然后与 1 比较
//     //    我们可用 fmin 来把这两种情况统一写出来
//     double limiter = 2.0 * slope / (1.0 + slope);   // 对应 "2R/(1+R)" 公式
//     double limited = fmin(slope, limiter);          // 若 slope>1 时 limited=limiter，否则= slope
//     return fmin(1.0, limited);                      // 再与 1.0 比较
// }
// __device__ double limiterR2(double smaller, double larger)
// {
//     // 1) 若 larger == 0.0，则根据原逻辑可知无论 smaller 为 0 与否都返回 0.0
//     //    (因为原先: if (smaller != 0 && larger==0) => return 0; if (smaller==0 && larger==0) => slope=0 => 0)
//     if (larger == 0.0) {
//         return 0.0;
//     }

//     // 2) 计算 slope
//     double slope = smaller / larger;

//     // 3) 若 slope <= 0，直接返回 0
//     if (slope <= 0.0) {
//         return 0.0;
//     }

//     // 4) 若 slope <= 1 => 结果即 slope
//     //    若 slope > 1  => 2/(1 + slope) 再与 1 比较
//     double limiter = 2.0 / (1.0 + slope);
//     double limited = fmin(slope, limiter);
//     return fmin(1.0, limited);
// }


__device__ void get_flux_x(const double *pri, double *flux) {
    flux[0] = pri[0]*pri[1];
    flux[1] = pri[0]*pri[1]*pri[1] + pri[3];
    flux[2] = pri[0]*pri[1]*pri[2];
    double Energy = 0.5*pri[0]*(pri[1]*pri[1] + pri[2]*pri[2]) + pri[3]/(1.4-1.0);
    flux[3] = pri[1]*(pri[3] + Energy);
}

__device__ void get_flux_y(const double *pri, double *flux) {
    flux[0] = pri[0]*pri[2];
    flux[1] = pri[0]*pri[1]*pri[2];
    flux[2] = pri[0]*pri[2]*pri[2] + pri[3];
    double Energy = 0.5*pri[0]*(pri[1]*pri[1] + pri[2]*pri[2]) + pri[3]/(1.4-1.0);
    flux[3] = pri[2]*(pri[3] + Energy);
}

__global__ void computeHalftimeKernel_x(
    const solVectors d_data_con,
    solVectors d_half_uL,
    solVectors d_half_uR, 
    double dt,
    double dx,
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
    
    // --- Step 1: 读取 co  n(i,j), con(i-1,j), con(i+1,j) ---
    double conM[4];  // con(i,j)
    double conL[4];  // con(i-1,j)
    double conR[4];  // con(i+1,j)
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
    double tempL[4], tempR[4];
    for (int k = 0; k < 4; k++) {
        double temp1 = conM[k] - conL[k];  // i - (i-1)
        double temp2 = conR[k] - conM[k];  // (i+1) - i
        double di = 0.5 * (temp1 + temp2);

        // 这里分别调用 limiterL2 / limiterR2：
        double phiL = limiterL2(temp1, temp2);
        double phiR = limiterR2(temp1, temp2);

        // 得到左右临时状态
        tempL[k] = conM[k] - 0.5 * di * phiL;
        tempR[k] = conM[k] + 0.5 * di * phiR;
    }
    // --- Step 3: 将 tempL, tempR 转为原始量 priL, priR，并计算通量 fluxL, fluxR ---
    double priL[4], priR[4];
    get_pri(tempL, priL);
    get_pri(tempR, priR);

    double fluxL[4], fluxR[4];
    get_flux_x(priL, fluxL);
    get_flux_x(priR, fluxR);
    
    // --- Step 4: 半步更新 (回到保守量空间) ---
    // tempL, tempR 各减去 0.5*(dt/dx)*(fluxR - fluxL)
    for (int k = 0; k < 4; k++) {
        double delta = 0.5 * (dt / dx) * (fluxR[k] - fluxL[k]);
        tempL[k] = tempL[k] - delta;
        tempR[k] = tempR[k] - delta;
    }

    // --- Step 5: 把结果存到 half_uL, half_uR 里 ---
    int out_idx = j*(nx + 2) + (i-1);
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

__global__ void computeHalftimeKernel_y(
    const solVectors d_data_con,
    solVectors d_half_uL,
    solVectors d_half_uR, 
    double dt,
    double dx,
    int nx, int ny
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= (nx+4) || j >= (ny+3)) {
        return;
    }
    int stride = (nx + 4);
    int idx      = j*stride + i;
    int idx_left = (j - 1)*stride + i;
    int idx_right= (j + 1)*stride + i;
    
    // --- Step 1: 读取 co  n(i,j), con(i-1,j), con(i+1,j) ---
    double conM[4];  // con(i,j)
    double conL[4];  // con(i-1,j)
    double conR[4];  // con(i+1,j)
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
    double tempL[4], tempR[4];
    for (int k = 0; k < 4; k++) {
        double temp1 = conM[k] - conL[k];  // i - (i-1)
        double temp2 = conR[k] - conM[k];  // (i+1) - i
        double di = 0.5 * (temp1 + temp2);

        // 这里分别调用 limiterL2 / limiterR2：
        double phiL = limiterL2(temp1, temp2);
        double phiR = limiterR2(temp1, temp2);

        // 得到左右临时状态
        tempL[k] = conM[k] - 0.5 * di * phiL;
        tempR[k] = conM[k] + 0.5 * di * phiR;
    }
    // --- Step 3: 将 tempL, tempR 转为原始量 priL, priR，并计算通量 fluxL, fluxR ---
    double priL[4], priR[4];
    get_pri(tempL, priL);
    get_pri(tempR, priR);

    double fluxL[4], fluxR[4];
    get_flux_y(priL, fluxL);
    get_flux_y(priR, fluxR);
    
    // --- Step 4: 半步更新 (回到保守量空间) ---
    // tempL, tempR 各减去 0.5*(dt/dx)*(fluxR - fluxL)
    for (int k = 0; k < 4; k++) {
        double delta = 0.5 * (dt / dx) * (fluxR[k] - fluxL[k]);
        tempL[k] = tempL[k] - delta;
        tempR[k] = tempR[k] - delta;
    }

    // --- Step 5: 把结果存到 half_uL, half_uR 里 ---
    int out_idx = (j-1)*(nx + 4) + i;
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


void computeHalftime(
    const solVectors &d_data_con,
    solVectors &d_half_uL,
    solVectors &d_half_uR,
    double dt,
    int choice
)
{
    if (choice == 1)
    {
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uL.rho), (nx+2) * (ny+4) * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uL.vx),  (nx+2) * (ny+4) * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uL.vy),  (nx+2) * (ny+4) * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uL.p),   (nx+2) * (ny+4) * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uR.rho), (nx+2) * (ny+4) * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uR.vx),  (nx+2) * (ny+4) * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uR.vy),  (nx+2) * (ny+4) * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uR.p),   (nx+2) * (ny+4) * sizeof(double)));

        dim3 block(16, 16);
        dim3 grid( (nx+block.x-1+4)/block.x, (ny+block.y-1+4)/block.y );
        computeHalftimeKernel_x<<<grid, block>>>(
            d_data_con,    
            d_half_uL,     
            d_half_uR,     
            dt, dx, 
            nx, ny
        );
    }
    else{
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uL.rho), (nx+4) * (ny+2) * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uL.vx),  (nx+4) * (ny+2) * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uL.vy),  (nx+4) * (ny+2) * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uL.p),   (nx+4) * (ny+2) * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uR.rho), (nx+4) * (ny+2) * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uR.vx),  (nx+4) * (ny+2) * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uR.vy),  (nx+4) * (ny+2) * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&(d_half_uR.p),   (nx+4) * (ny+2) * sizeof(double)));

        dim3 block(16, 16);
        dim3 grid( (nx+block.x-1+4)/block.x, (ny+block.y-1+4)/block.y );
        computeHalftimeKernel_y<<<grid, block>>>(
            d_data_con,    
            d_half_uL,     
            d_half_uR,     
            dt, dx, 
            nx, ny
        );
    }
}

// ---------------------- GPU 内核：计算 x 方向 SLIC flux ----------------------
__global__ void computeSLICFluxKernel_x(
    const solVectors d_half_uL,  // 左侧半步状态，尺寸： (nx+2) x (ny+4)
    const solVectors d_half_uR,  // 右侧半步状态，同上
    solVectors d_SLIC_flux,      // 输出 SLIC flux，尺寸： (nx-3) x ny
    double dt,
    double dx,
    int nx,  int ny
)
{
    // 设定输出 SLIC flux 的二维域：
    // i_flux 范围： 0 <= i < (nx - 3)
    // j_flux 范围： 0 <= j < ny
    int i_flux = blockIdx.x * blockDim.x + threadIdx.x;
    int j_flux = blockIdx.y * blockDim.y + threadIdx.y;
    if (i_flux >= (nx + 1) || j_flux >= ny + 4)
        return;
    int half_width = nx + 2;  // 水平步长
    int index_L = j_flux * half_width + (i_flux + 1);  // d_half_uL 对应元素
    int index_R = j_flux * half_width + (i_flux);      // d_half_uR 对应元素
    int flux_idx  = j_flux * (nx+1) + i_flux;          // 输出 SLIC flux 的线性索引
    // 读取半步状态（保守量格式）：每个状态有4个分量
    double consL[4], consR[4];
    consL[0] = d_half_uL.rho[index_L];
    consL[1] = d_half_uL.vx[index_L];
    consL[2] = d_half_uL.vy[index_L];
    consL[3] = d_half_uL.p[index_L];

    consR[0] = d_half_uR.rho[index_R];
    consR[1] = d_half_uR.vx[index_R];
    consR[2] = d_half_uR.vy[index_R];
    consR[3] = d_half_uR.p[index_R];

    // ---------------- Step 1: 转换为原始量，并计算 x 方向通量 ----------------
    double priL[4], priR[4];
    get_pri(consL, priL);
    get_pri(consR, priR);

    double fluxL[4], fluxR[4];
    get_flux_x(priL, fluxL);
    get_flux_x(priR, fluxR);

    // ---------------- Step 2: 计算 LF 与 RI_U ----------------
    double LF[4], RI_U[4];
    for (int k = 0; k < 4; k++) {
        LF[k]   = 0.5 * (fluxL[k] + fluxR[k]) + 0.5 * (dx / dt) * (consR[k] - consL[k]);
        RI_U[k] = 0.5 * (consL[k] + consR[k]) - 0.5 * (dt / dx) * (fluxL[k] - fluxR[k]);
    }

    // ---------------- Step 3: 计算 RI 通量 ----------------
    double pri_RI[4], RI[4];
    get_pri(RI_U, pri_RI);
    get_flux_x(pri_RI, RI);
    // ---------------- Step 4: 计算最终 SLIC flux = 0.5*(LF + RI) ----------------
    double slic_flux[4];
    for (int k = 0; k < 4; k++) {
        slic_flux[k] = 0.5 * (LF[k] + RI[k]);
    }
    d_SLIC_flux.rho[flux_idx] = slic_flux[0];
    d_SLIC_flux.vx [flux_idx] = slic_flux[1];
    d_SLIC_flux.vy [flux_idx] = slic_flux[2];
    d_SLIC_flux.p  [flux_idx] = slic_flux[3];
}

__global__ void computeSLICFluxKernel_y(
    const solVectors d_half_uL,  // 左侧半步状态，尺寸： (nx+2) x (ny+4)
    const solVectors d_half_uR,  // 右侧半步状态，同上
    solVectors d_SLIC_flux,      // 输出 SLIC flux，尺寸： (nx-3) x ny
    double dt,
    double dx,
    int nx,  int ny
)
{
    // 设定输出 SLIC flux 的二维域：
    // i_flux 范围： 0 <= i < (nx - 3)
    // j_flux 范围： 0 <= j < ny
    int i_flux = blockIdx.x * blockDim.x + threadIdx.x;
    int j_flux = blockIdx.y * blockDim.y + threadIdx.y;
    if (i_flux >= (nx + 4) || j_flux >= ny + 1)
        return;
    int half_width = nx + 4;  // 水平步长
    int index_L = (j_flux + 1) * half_width + i_flux;  // d_half_uL 对应元素
    int index_R = j_flux * half_width + i_flux;      // d_half_uR 对应元素
    int flux_idx  = j_flux * half_width + i_flux;          // 输出 SLIC flux 的线性索引
    // 读取半步状态（保守量格式）：每个状态有4个分量
    double consL[4], consR[4];
    consL[0] = d_half_uL.rho[index_L];
    consL[1] = d_half_uL.vx[index_L];
    consL[2] = d_half_uL.vy[index_L];
    consL[3] = d_half_uL.p[index_L];

    consR[0] = d_half_uR.rho[index_R];
    consR[1] = d_half_uR.vx[index_R];
    consR[2] = d_half_uR.vy[index_R];
    consR[3] = d_half_uR.p[index_R];

    // ---------------- Step 1: 转换为原始量，并计算 x 方向通量 ----------------
    double priL[4], priR[4];
    get_pri(consL, priL);
    get_pri(consR, priR);

    double fluxL[4], fluxR[4];
    get_flux_y(priL, fluxL);
    get_flux_y(priR, fluxR);

    // ---------------- Step 2: 计算 LF 与 RI_U ----------------
    double LF[4], RI_U[4];
    for (int k = 0; k < 4; k++) {
        LF[k]   = 0.5 * (fluxL[k] + fluxR[k]) + 0.5 * (dx / dt) * (consR[k] - consL[k]);
        RI_U[k] = 0.5 * (consL[k] + consR[k]) - 0.5 * (dt / dx) * (fluxL[k] - fluxR[k]);
    }

    // ---------------- Step 3: 计算 RI 通量 ----------------
    double pri_RI[4], RI[4];
    get_pri(RI_U, pri_RI);
    get_flux_y(pri_RI, RI);
    // ---------------- Step 4: 计算最终 SLIC flux = 0.5*(LF + RI) ----------------
    double slic_flux[4];
    for (int k = 0; k < 4; k++) {
        slic_flux[k] = 0.5 * (LF[k] + RI[k]);
    }
    d_SLIC_flux.rho[flux_idx] = slic_flux[0];
    d_SLIC_flux.vx [flux_idx] = slic_flux[1];
    d_SLIC_flux.vy [flux_idx] = slic_flux[2];
    d_SLIC_flux.p  [flux_idx] = slic_flux[3];
}


void computeSLICFlux(
    const solVectors &d_half_uL,
    const solVectors &d_half_uR,
    solVectors &d_SLIC_flux,  // 输出：SLIC flux
    double dt,
    int choice 
)
{
    if (choice == 1)
    {
        CUDA_CHECK(cudaMalloc((void**)&(d_SLIC_flux.rho), (nx + 1) * (ny+4) * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&(d_SLIC_flux.vx),  (nx + 1) * (ny+4) * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&(d_SLIC_flux.vy),  (nx + 1) * (ny+4) * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&(d_SLIC_flux.p),   (nx + 1) * (ny+4) * sizeof(double)));

        dim3 block(16, 16);
        dim3 grid( ((nx + 1) + block.x - 1) / block.x, (ny + 4 + block.y - 1) / block.y );
        computeSLICFluxKernel_x<<<grid, block>>>(
            d_half_uL,
            d_half_uR,
            d_SLIC_flux,
            dt,
            dx,
            nx,
            ny
        );
    }
    else{
        CUDA_CHECK(cudaMalloc((void**)&(d_SLIC_flux.rho), (nx + 4) * (ny+1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&(d_SLIC_flux.vx),  (nx + 4) * (ny+1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&(d_SLIC_flux.vy),  (nx + 4) * (ny+1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&(d_SLIC_flux.p),   (nx + 4) * (ny+1) * sizeof(double)));

        dim3 block(16, 16);
        dim3 grid( ((nx + 4) + block.x - 1) / block.x, (ny + 1 + block.y - 1) / block.y );
        computeSLICFluxKernel_y<<<grid, block>>>(
            d_half_uL,
            d_half_uR,
            d_SLIC_flux,
            dt,
            dx,
            nx,
            ny
        );
    }
}

__global__ void updateKernel_x(
    solVectors d_data_con,           // 待更新的状态，尺寸为 (nx+4) x (ny+4)
    const solVectors d_SLIC_flux, // SLIC flux 数组，尺寸为 (nx-3) x ny
    double dt,
    double dx,
    int nx,                     // 原问题的网格数（不含 ghost），例如 CPU 中 old_u.size() 的横向部分
    int ny
){
    int i_upd = blockIdx.x * blockDim.x + threadIdx.x;
    int j     = blockIdx.y * blockDim.y + threadIdx.y;
    if (i_upd >= nx || j >= (ny+4))
        return;
    int stride_old = nx + 4;
    int idx = j * stride_old + (i_upd + 2);

    int stride_flux = nx + 1;
    int flux_idx1 = j * stride_flux + i_upd;
    int flux_idx2 = j * stride_flux + (i_upd + 1);

    d_data_con.rho[idx] = d_data_con.rho[idx] - (dt/dx) * (d_SLIC_flux.rho[flux_idx2] - d_SLIC_flux.rho[flux_idx1]);
    d_data_con.vx[idx]  = d_data_con.vx[idx]  - (dt/dx) * (d_SLIC_flux.vx[flux_idx2]  - d_SLIC_flux.vx[flux_idx1]);
    d_data_con.vy[idx]  = d_data_con.vy[idx]  - (dt/dx) * (d_SLIC_flux.vy[flux_idx2]  - d_SLIC_flux.vy[flux_idx1]);
    d_data_con.p[idx]   = d_data_con.p[idx]   - (dt/dx) * (d_SLIC_flux.p[flux_idx2]   - d_SLIC_flux.p[flux_idx1]);
}

__global__ void updateKernel_y(
    solVectors d_data_con,           // 待更新的状态，尺寸为 (nx+4) x (ny+4)
    const solVectors d_SLIC_flux, // SLIC flux 数组，尺寸为 (nx-3) x ny
    double dt,
    double dx,
    int nx,                     // 原问题的网格数（不含 ghost），例如 CPU 中 old_u.size() 的横向部分
    int ny
){
    int i_upd = blockIdx.x * blockDim.x + threadIdx.x;
    int j     = blockIdx.y * blockDim.y + threadIdx.y;
    if (i_upd >= nx + 4 || j >= ny)
        return;
    int stride_old = nx + 4;
    int idx = (j + 2) * stride_old + i_upd;

    
    int flux_idx1 = j * stride_old + i_upd;
    int flux_idx2 = (j + 1) * stride_old + i_upd;

    d_data_con.rho[idx] = d_data_con.rho[idx] - (dt/dx) * (d_SLIC_flux.rho[flux_idx2] - d_SLIC_flux.rho[flux_idx1]);
    d_data_con.vx[idx]  = d_data_con.vx[idx]  - (dt/dx) * (d_SLIC_flux.vx[flux_idx2]  - d_SLIC_flux.vx[flux_idx1]);
    d_data_con.vy[idx]  = d_data_con.vy[idx]  - (dt/dx) * (d_SLIC_flux.vy[flux_idx2]  - d_SLIC_flux.vy[flux_idx1]);
    d_data_con.p[idx]   = d_data_con.p[idx]   - (dt/dx) * (d_SLIC_flux.p[flux_idx2]   - d_SLIC_flux.p[flux_idx1]);
}


void updateSolution(
    solVectors &d_data_con,
    const solVectors &d_SLIC_flux,
    double dt,
    int choice
)
{
    if (choice == 1)
    {
    dim3 block(16, 16);
    dim3 grid( (nx + block.x - 1) / block.x,
               (ny + 4 + block.y - 1) / block.y );
    updateKernel_x<<<grid, block>>>(d_data_con, d_SLIC_flux, dt, dx, nx, ny);
    }
    else{
    dim3 block(16, 16);
    dim3 grid( (nx + 4 + block.x - 1) / block.x,
               (ny + block.y - 1) / block.y );
    updateKernel_y<<<grid, block>>>(d_data_con, d_SLIC_flux, dt, dx, nx, ny);
    }
}


__global__ void list_con2priKernel(
    const solVectors d_data_con,
    solVectors d_data_pri,
    int nx, int ny
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= (nx+4) || j >= (ny+4)) {
        return;
    }
    int stride = (nx + 4);
    int idx = j*stride + i;
    double con[4];
    con[0] = d_data_con.rho[idx];
    con[1] = d_data_con.vx [idx];
    con[2] = d_data_con.vy [idx];
    con[3] = d_data_con.p  [idx];
    double pri[4];
    get_pri(con, pri);
    d_data_pri.rho[idx] = pri[0];
    d_data_pri.vx [idx] = pri[1];
    d_data_pri.vy [idx] = pri[2];
    d_data_pri.p  [idx] = pri[3];
}

void list_con2pri(
    solVectors &d_data_con,
    solVectors &d_data_pri
)
{
    CUDA_CHECK(cudaFree(d_data_pri.rho));
    CUDA_CHECK(cudaFree(d_data_pri.vx));
    CUDA_CHECK(cudaFree(d_data_pri.vy));
    CUDA_CHECK(cudaFree(d_data_pri.p));
    CUDA_CHECK(cudaMalloc((void**)&(d_data_pri.rho), (nx+4) * (ny+4) * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&(d_data_pri.vx),  (nx+4) * (ny+4) * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&(d_data_pri.vy),  (nx+4) * (ny+4) * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&(d_data_pri.p),   (nx+4) * (ny+4) * sizeof(double)));

    dim3 block(16, 16);
    dim3 grid( (nx + 4 + block.x - 1) / block.x,
               (ny + 4 + block.y - 1) / block.y );
    list_con2priKernel<<<grid, block>>>(d_data_con, d_data_pri, nx, ny);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
            std::cerr << "CUDA kernel launch failed in list_con2pri: " 
                      << cudaGetErrorString(err) << std::endl;
            exit(-1);
        }
}

void freeDeviceMemory2(solVectors &d_half_uL, solVectors &d_half_uR, solVectors &d_SLIC_flux) {
    CUDA_CHECK(cudaFree(d_half_uL.rho));
    CUDA_CHECK(cudaFree(d_half_uL.vx));
    CUDA_CHECK(cudaFree(d_half_uL.vy));
    CUDA_CHECK(cudaFree(d_half_uL.p));
    CUDA_CHECK(cudaFree(d_half_uR.rho));
    CUDA_CHECK(cudaFree(d_half_uR.vx));
    CUDA_CHECK(cudaFree(d_half_uR.vy));
    CUDA_CHECK(cudaFree(d_half_uR.p));
    CUDA_CHECK(cudaFree(d_SLIC_flux.rho));
    CUDA_CHECK(cudaFree(d_SLIC_flux.vx));
    CUDA_CHECK(cudaFree(d_SLIC_flux.vy));
    CUDA_CHECK(cudaFree(d_SLIC_flux.p));
}

void store_data(const std::vector<double> rho, const std::vector<double> vx, const std::vector<double> vy, const std::vector<double> p, const double t, int step) {
    std::ostringstream filename;
  filename << "data/step_" << std::setw(4) << std::setfill('0') << step
           << ".csv";
  std::ofstream file(filename.str());
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename.str() << std::endl;
    return;
  }
  // 写入时间信息
  file << "# Time: " << t << "\n";
  // 写入数据
  for (int j = 2; j < ny+2; j++) {
    for (int i = 2; i < nx+2; i++) {
        int idx = j * (nx + 4) + i;
      file << rho[idx] << "," << vx[idx] << "," << vy[idx] << "," << p[idx];
      if (i < nx + 1) {
        file << ",";
      }
    }
    file << "\n";
  }
  file.close();
}

__global__ void compute_x_shared (
        solVectors d_data_con,
        double dt,
        double dx,
        int nx,
        int ny)
    {
        __shared__ double temp1_rho[BDIMX_Y][BDIMX_X];
        __shared__ double temp1_vx [BDIMX_Y][BDIMX_X];
        __shared__ double temp1_vy [BDIMX_Y][BDIMX_X];
        __shared__ double temp1_p  [BDIMX_Y][BDIMX_X];

        __shared__ double temp2_rho[BDIMX_Y][BDIMX_X-2];
        __shared__ double temp2_vx [BDIMX_Y][BDIMX_X-2];
        __shared__ double temp2_vy [BDIMX_Y][BDIMX_X-2];
        __shared__ double temp2_p  [BDIMX_Y][BDIMX_X-2];



        int iglobal = (blockIdx.x == 0) ? threadIdx.x : (BDIMX_X) + (BDIMX_X - 4) * (blockIdx.x - 1) + threadIdx.x-4;
        int jglobal = blockIdx.y * BDIMX_Y + threadIdx.y;
        int stride = nx + 4;
        if (iglobal >= nx + 4 || jglobal >= ny + 4) {
            return;
        }
        int idx = jglobal * stride + iglobal;
        temp1_rho[threadIdx.y][threadIdx.x] = d_data_con.rho[idx];
        temp1_vx [threadIdx.y][threadIdx.x] = d_data_con.vx [idx];
        temp1_vy [threadIdx.y][threadIdx.x] = d_data_con.vy [idx];
        temp1_p  [threadIdx.y][threadIdx.x] = d_data_con.p  [idx];
        __syncthreads();
        if (threadIdx.x < BDIMX_X - 2  && threadIdx.y < BDIMX_Y && iglobal < nx + 2 && jglobal < ny + 4) {
            int tempx = threadIdx.x + 1;
            double conM[4];  // con(i,j)
            double conL[4];  // con(i-1,j)
            double conR[4];  // con(i+1,j)
            conM[0] = temp1_rho[threadIdx.y][tempx];
            conM[1] = temp1_vx [threadIdx.y][tempx];  // 这里 vx 里实际存的是 rho*u
            conM[2] = temp1_vy [threadIdx.y][tempx];  // 这里 vy 里实际存的是 rho*v
            conM[3] = temp1_p  [threadIdx.y][tempx];  // E (总能量)

            conL[0] = temp1_rho[threadIdx.y][tempx - 1];
            conL[1] = temp1_vx [threadIdx.y][tempx - 1];
            conL[2] = temp1_vy [threadIdx.y][tempx - 1];
            conL[3] = temp1_p  [threadIdx.y][tempx - 1];

            conR[0] = temp1_rho[threadIdx.y][tempx + 1];
            conR[1] = temp1_vx [threadIdx.y][tempx + 1];
            conR[2] = temp1_vy [threadIdx.y][tempx + 1];
            conR[3] = temp1_p  [threadIdx.y][tempx + 1];
            // --- Step 2: 斜率限制，得到 tempL, tempR (仍在保守量空间) ---
            double tempL[4], tempR[4];
            for (int k = 0; k < 4; k++) {
                double temp1 = conM[k] - conL[k];  // i - (i-1)
                double temp2 = conR[k] - conM[k];  // (i+1) - i
                double di = 0.5 * (temp1 + temp2);

                // 这里分别调用 limiterL2 / limiterR2：
                double phiL = limiterL2(temp1, temp2);
                double phiR = limiterR2(temp1, temp2);

                // 得到左右临时状态
                tempL[k] = conM[k] - 0.5 * di * phiL;
                tempR[k] = conM[k] + 0.5 * di * phiR;
            }
            // --- Step 3: 将 tempL, tempR 转为原始量 priL, priR，并计算通量 fluxL, fluxR ---
            double priL[4], priR[4];
            get_pri(tempL, priL);
            get_pri(tempR, priR);

            double fluxL[4], fluxR[4];
            get_flux_x(priL, fluxL);
            get_flux_x(priR, fluxR);

            // --- Step 4: 半步更新 (回到保守量空间) ---
            // tempL, tempR 各减去 0.5*(dt/dx)*(fluxR - fluxL)
            for (int k = 0; k < 4; k++) {
                double delta = 0.5 * (dt / dx) * (fluxR[k] - fluxL[k]);
                tempL[k] = tempL[k] - delta;
                tempR[k] = tempR[k] - delta;
            }
            temp1_rho[threadIdx.y][threadIdx.x] = tempL[0];
            temp1_vx [threadIdx.y][threadIdx.x] = tempL[1];
            temp1_vy [threadIdx.y][threadIdx.x] = tempL[2];
            temp1_p  [threadIdx.y][threadIdx.x] = tempL[3];

            temp2_rho[threadIdx.y][threadIdx.x] = tempR[0];
            temp2_vx [threadIdx.y][threadIdx.x] = tempR[1];
            temp2_vy [threadIdx.y][threadIdx.x] = tempR[2];
            temp2_p  [threadIdx.y][threadIdx.x] = tempR[3];
            
        }
        __syncthreads();
        if(threadIdx.x < BDIMX_X - 3 && iglobal < nx + 1 && jglobal < ny + 4){
            int index_XL = threadIdx.x + 1;
            int index_XR = threadIdx.x;
            int index_Y  = threadIdx.y;

            double consL[4], consR[4];
            consL[0] = temp1_rho[index_Y][index_XL];
            consL[1] = temp1_vx [index_Y][index_XL];
            consL[2] = temp1_vy [index_Y][index_XL];
            consL[3] = temp1_p  [index_Y][index_XL];

            consR[0] = temp2_rho[index_Y][index_XR];
            consR[1] = temp2_vx [index_Y][index_XR];
            consR[2] = temp2_vy [index_Y][index_XR];
            consR[3] = temp2_p  [index_Y][index_XR];

            // ---------------- Step 1: 转换为原始量，并计算 x 方向通量 ----------------
            double priL[4], priR[4];
            get_pri(consL, priL);
            get_pri(consR, priR);

            double fluxL[4], fluxR[4];
            get_flux_x(priL, fluxL);
            get_flux_x(priR, fluxR);

            // ---------------- Step 2: 计算 LF 与 RI_U ----------------
            double LF[4], RI_U[4];
            for (int k = 0; k < 4; k++) {
                LF[k]   = 0.5 * (fluxL[k] + fluxR[k]) + 0.5 * (dx / dt) * (consR[k] - consL[k]);
                RI_U[k] = 0.5 * (consL[k] + consR[k]) - 0.5 * (dt / dx) * (fluxL[k] - fluxR[k]);
            }
            // ---------------- Step 3: 计算 RI 通量 ----------------
            double pri_RI[4], RI[4];
            get_pri(RI_U, pri_RI);
            get_flux_x(pri_RI, RI);
            // ---------------- Step 4: 计算最终 SLIC flux = 0.5*(LF + RI) ----------------
            double slic_flux[4];
            for (int k = 0; k < 4; k++) {
                slic_flux[k] = 0.5 * (LF[k] + RI[k]);
            }
            temp1_rho[threadIdx.y][threadIdx.x] = slic_flux[0];
            temp1_vx [threadIdx.y][threadIdx.x] = slic_flux[1];
            temp1_vy [threadIdx.y][threadIdx.x] = slic_flux[2];
            temp1_p  [threadIdx.y][threadIdx.x] = slic_flux[3];
        }
        __syncthreads();
        // start to update the data
        if (threadIdx.x < BDIMX_X - 4 && iglobal < nx && jglobal < ny + 4) {
            int stride_old = nx + 4;
            int idx = jglobal * stride_old + (iglobal + 2);
            d_data_con.rho[idx] = d_data_con.rho[idx] - (dt/dx) * (temp1_rho[threadIdx.y][threadIdx.x + 1] - temp1_rho[threadIdx.y][threadIdx.x]);
            d_data_con.vx[idx]  = d_data_con.vx[idx]  - (dt/dx) * (temp1_vx [threadIdx.y][threadIdx.x + 1] - temp1_vx [threadIdx.y][threadIdx.x]);
            d_data_con.vy[idx]  = d_data_con.vy[idx]  - (dt/dx) * (temp1_vy [threadIdx.y][threadIdx.x + 1] - temp1_vy [threadIdx.y][threadIdx.x]);
            d_data_con.p[idx]   = d_data_con.p[idx]   - (dt/dx) * (temp1_p  [threadIdx.y][threadIdx.x + 1] - temp1_p  [threadIdx.y][threadIdx.x]);
        }
        __syncthreads();
    }

__global__ void compute_y_shared (
        solVectors d_data_con,
        double dt,
        double dy,
        int nx,
        int ny)
    {

        __shared__ double temp1_rho[BDIMY_Y][BDIMY_X];
        __shared__ double temp1_vx [BDIMY_Y][BDIMY_X];
        __shared__ double temp1_vy [BDIMY_Y][BDIMY_X];
        __shared__ double temp1_p  [BDIMY_Y][BDIMY_X];

        __shared__ double temp2_rho[BDIMY_Y-2][BDIMY_X];
        __shared__ double temp2_vx [BDIMY_Y-2][BDIMY_X];
        __shared__ double temp2_vy [BDIMY_Y-2][BDIMY_X];
        __shared__ double temp2_p  [BDIMY_Y-2][BDIMY_X];

        int iglobal = blockIdx.x * blockDim.x + threadIdx.x;
        int jglobal = (blockIdx.y == 0) ? threadIdx.y : (BDIMY_Y) + (BDIMY_Y - 4) * (blockIdx.y - 1) + threadIdx.y-4;
        int stride = nx + 4;
        if (iglobal >= nx + 4 || jglobal >= ny + 4) {
            return;
        }
        int idx = jglobal * stride + iglobal;
        temp1_rho[threadIdx.y][threadIdx.x] = d_data_con.rho[idx];
        temp1_vx [threadIdx.y][threadIdx.x] = d_data_con.vx [idx];
        temp1_vy [threadIdx.y][threadIdx.x] = d_data_con.vy [idx];
        temp1_p  [threadIdx.y][threadIdx.x] = d_data_con.p  [idx];
        __syncthreads();
        if (threadIdx.y < BDIMY_Y - 2  && threadIdx.x < BDIMY_X && iglobal < nx + 4 && jglobal < ny + 2) {
            int tempy = threadIdx.y + 1;
            double conM[4];  // con(i,j)
            double conL[4];  // con(i-1,j)
            double conR[4];  // con(i+1,j)
            conM[0] = temp1_rho[tempy][threadIdx.x];
            conM[1] = temp1_vx [tempy][threadIdx.x];  // 这里 vx 里实际存的是 rho*u
            conM[2] = temp1_vy [tempy][threadIdx.x];  // 这里 vy 里实际存的是 rho*v
            conM[3] = temp1_p  [tempy][threadIdx.x];  // E (总能量)

            conL[0] = temp1_rho[tempy - 1][threadIdx.x];
            conL[1] = temp1_vx [tempy - 1][threadIdx.x];
            conL[2] = temp1_vy [tempy - 1][threadIdx.x];
            conL[3] = temp1_p  [tempy - 1][threadIdx.x];

            conR[0] = temp1_rho[tempy + 1][threadIdx.x];
            conR[1] = temp1_vx [tempy + 1][threadIdx.x];
            conR[2] = temp1_vy [tempy + 1][threadIdx.x];
            conR[3] = temp1_p  [tempy + 1][threadIdx.x];
            // --- Step 2: 斜率限制，得到 tempL, tempR (仍在保守量空间) ---
            double tempL[4], tempR[4];
            for (int k = 0; k < 4; k++) {
                double temp1 = conM[k] - conL[k];  // i - (i-1)
                double temp2 = conR[k] - conM[k];  // (i+1) - i
                double di = 0.5 * (temp1 + temp2);

                // 这里分别调用 limiterL2 / limiterR2：
                double phiL = limiterL2(temp1, temp2);
                double phiR = limiterR2(temp1, temp2);

                // 得到左右临时状态
                tempL[k] = conM[k] - 0.5 * di * phiL;
                tempR[k] = conM[k] + 0.5 * di * phiR;
            }
            // --- Step 3: 将 tempL, tempR 转为原始量 priL, priR，并计算通量 fluxL, fluxR ---
            double priL[4], priR[4];
            get_pri(tempL, priL);
            get_pri(tempR, priR);

            double fluxL[4], fluxR[4];
            get_flux_y(priL, fluxL);
            get_flux_y(priR, fluxR);

            // --- Step 4: 半步更新 (回到保守量空间) ---
            // tempL, tempR 各减去 0.5*(dt/dx)*(fluxR - fluxL)
            for (int k = 0; k < 4; k++) {
                double delta = 0.5 * (dt / dy) * (fluxR[k] - fluxL[k]);
                tempL[k] = tempL[k] - delta;
                tempR[k] = tempR[k] - delta;
            }
            temp1_rho[threadIdx.y][threadIdx.x] = tempL[0];
            temp1_vx [threadIdx.y][threadIdx.x] = tempL[1];
            temp1_vy [threadIdx.y][threadIdx.x] = tempL[2];
            temp1_p  [threadIdx.y][threadIdx.x] = tempL[3];

            temp2_rho[threadIdx.y][threadIdx.x] = tempR[0];
            temp2_vx [threadIdx.y][threadIdx.x] = tempR[1];
            temp2_vy [threadIdx.y][threadIdx.x] = tempR[2];
            temp2_p  [threadIdx.y][threadIdx.x] = tempR[3];
            
        }
        __syncthreads();
        if(threadIdx.y < BDIMY_Y - 3 && iglobal < nx + 4 && jglobal < ny + 1){
            int index_YL = threadIdx.y + 1;
            int index_YR = threadIdx.y;
            int index_X  = threadIdx.x;

            double consL[4], consR[4];
            consL[0] = temp1_rho[index_YL][index_X];
            consL[1] = temp1_vx[index_YL][index_X];
            consL[2] = temp1_vy[index_YL][index_X];
            consL[3] = temp1_p[index_YL][index_X];

            consR[0] = temp2_rho[index_YR][index_X];
            consR[1] = temp2_vx[index_YR][index_X];
            consR[2] = temp2_vy[index_YR][index_X];
            consR[3] = temp2_p [index_YR][index_X];

            // ---------------- Step 1: 转换为原始量，并计算 x 方向通量 ----------------
            double priL[4], priR[4];
            get_pri(consL, priL);
            get_pri(consR, priR);

            double fluxL[4], fluxR[4];
            get_flux_y(priL, fluxL);
            get_flux_y(priR, fluxR);

            // ---------------- Step 2: 计算 LF 与 RI_U ----------------
            double LF[4], RI_U[4];
            for (int k = 0; k < 4; k++) {
                LF[k]   = 0.5 * (fluxL[k] + fluxR[k]) + 0.5 * (dy / dt) * (consR[k] - consL[k]);
                RI_U[k] = 0.5 * (consL[k] + consR[k]) - 0.5 * (dt / dy) * (fluxL[k] - fluxR[k]);
            }
            // ---------------- Step 3: 计算 RI 通量 ----------------
            double pri_RI[4], RI[4];
            get_pri(RI_U, pri_RI);
            get_flux_y(pri_RI, RI);
            // ---------------- Step 4: 计算最终 SLIC flux = 0.5*(LF + RI) ----------------
            double slic_flux[4];
            for (int k = 0; k < 4; k++) {
                slic_flux[k] = 0.5 * (LF[k] + RI[k]);
            }
            temp1_rho[threadIdx.y][threadIdx.x] = slic_flux[0];
            temp1_vx[threadIdx.y][threadIdx.x] = slic_flux[1];
            temp1_vy[threadIdx.y][threadIdx.x] = slic_flux[2];
            temp1_p[threadIdx.y][threadIdx.x] = slic_flux[3];
        }
        __syncthreads();
        // start to update the data
        if (threadIdx.y < BDIMY_Y - 4 && iglobal < nx + 4 && jglobal < ny) {
            int stride_old = nx + 4;
            int idx = (jglobal+2) * stride_old + iglobal;
            d_data_con.rho[idx] = d_data_con.rho[idx] - (dt/dy) * (temp1_rho[threadIdx.y+1][threadIdx.x] - temp1_rho[threadIdx.y][threadIdx.x]);
            d_data_con.vx[idx]  = d_data_con.vx[idx]  - (dt/dy) * (temp1_vx [threadIdx.y+1][threadIdx.x] - temp1_vx [threadIdx.y][threadIdx.x]);
            d_data_con.vy[idx]  = d_data_con.vy[idx]  - (dt/dy) * (temp1_vy [threadIdx.y+1][threadIdx.x] - temp1_vy [threadIdx.y][threadIdx.x]);
            d_data_con.p[idx]   = d_data_con.p[idx]   - (dt/dy) * (temp1_p  [threadIdx.y+1][threadIdx.x] - temp1_p  [threadIdx.y][threadIdx.x]);
        }
        __syncthreads();
    }



__global__ void compute_shared (
        solVectors d_data_con,
        double dt,
        double dx,
        int nx,
        int ny)
    {
        __shared__ double temp1_rho[BDIMY_Y][BDIMX_X];
        __shared__ double temp1_vx [BDIMY_Y][BDIMX_X];
        __shared__ double temp1_vy [BDIMY_Y][BDIMX_X];
        __shared__ double temp1_p  [BDIMY_Y][BDIMX_X];

        __shared__ double temp2_rho[BDIMY_Y][BDIMX_X];
        __shared__ double temp2_vx [BDIMY_Y][BDIMX_X];
        __shared__ double temp2_vy [BDIMY_Y][BDIMX_X];
        __shared__ double temp2_p  [BDIMY_Y][BDIMX_X];

        __shared__ double temp3_rho[BDIMY_Y][BDIMX_X];
        __shared__ double temp3_vx [BDIMY_Y][BDIMX_X];
        __shared__ double temp3_vy [BDIMY_Y][BDIMX_X];
        __shared__ double temp3_p  [BDIMY_Y][BDIMX_X];



        int iglobal = (blockIdx.x == 0) ? threadIdx.x : (BDIMX_X) + (BDIMX_X - 4) * (blockIdx.x - 1) + threadIdx.x-4;
        int jglobal = (blockIdx.y == 0) ? threadIdx.y : (BDIMY_Y) + (BDIMY_Y - 4) * (blockIdx.y - 1) + threadIdx.y-4;
        int stride = nx + 4;
        if (iglobal >= nx + 4 || jglobal >= ny + 4) {
            return;
        }
        int idx = jglobal * stride + iglobal;
        temp1_rho[threadIdx.y][threadIdx.x] = d_data_con.rho[idx];
        temp1_vx [threadIdx.y][threadIdx.x] = d_data_con.vx [idx];
        temp1_vy [threadIdx.y][threadIdx.x] = d_data_con.vy [idx];
        temp1_p  [threadIdx.y][threadIdx.x] = d_data_con.p  [idx];
        __syncthreads();
        if (threadIdx.x < BDIMX_X - 2  && threadIdx.y < BDIMY_Y && iglobal < nx + 2 && jglobal < ny + 4) {
            int tempx = threadIdx.x + 1;
            double conM[4];  // con(i,j)
            double conL[4];  // con(i-1,j)
            double conR[4];  // con(i+1,j)
            conM[0] = temp1_rho[threadIdx.y][tempx];
            conM[1] = temp1_vx [threadIdx.y][tempx];  // 这里 vx 里实际存的是 rho*u
            conM[2] = temp1_vy [threadIdx.y][tempx];  // 这里 vy 里实际存的是 rho*v
            conM[3] = temp1_p  [threadIdx.y][tempx];  // E (总能量)

            conL[0] = temp1_rho[threadIdx.y][tempx - 1];
            conL[1] = temp1_vx [threadIdx.y][tempx - 1];
            conL[2] = temp1_vy [threadIdx.y][tempx - 1];
            conL[3] = temp1_p  [threadIdx.y][tempx - 1];

            conR[0] = temp1_rho[threadIdx.y][tempx + 1];
            conR[1] = temp1_vx [threadIdx.y][tempx + 1];
            conR[2] = temp1_vy [threadIdx.y][tempx + 1];
            conR[3] = temp1_p  [threadIdx.y][tempx + 1];
            // --- Step 2: 斜率限制，得到 tempL, tempR (仍在保守量空间) ---
            double tempL[4], tempR[4];
            for (int k = 0; k < 4; k++) {
                double temp1 = conM[k] - conL[k];  // i - (i-1)
                double temp2 = conR[k] - conM[k];  // (i+1) - i
                double di = 0.5 * (temp1 + temp2);

                // 这里分别调用 limiterL2 / limiterR2：
                double phiL = limiterL2(temp1, temp2);
                double phiR = limiterR2(temp1, temp2);

                // 得到左右临时状态
                tempL[k] = conM[k] - 0.5 * di * phiL;
                tempR[k] = conM[k] + 0.5 * di * phiR;
            }
            // --- Step 3: 将 tempL, tempR 转为原始量 priL, priR，并计算通量 fluxL, fluxR ---
            double priL[4], priR[4];
            get_pri(tempL, priL);
            get_pri(tempR, priR);

            double fluxL[4], fluxR[4];
            get_flux_x(priL, fluxL);
            get_flux_x(priR, fluxR);

            // --- Step 4: 半步更新 (回到保守量空间) ---
            // tempL, tempR 各减去 0.5*(dt/dx)*(fluxR - fluxL)
            for (int k = 0; k < 4; k++) {
                double delta = 0.5 * (dt / dx) * (fluxR[k] - fluxL[k]);
                tempL[k] = tempL[k] - delta;
                tempR[k] = tempR[k] - delta;
            }
            temp3_rho[threadIdx.y][threadIdx.x] = tempL[0];
            temp3_vx [threadIdx.y][threadIdx.x] = tempL[1];
            temp3_vy [threadIdx.y][threadIdx.x] = tempL[2];
            temp3_p  [threadIdx.y][threadIdx.x] = tempL[3];

            temp2_rho[threadIdx.y][threadIdx.x] = tempR[0];
            temp2_vx [threadIdx.y][threadIdx.x] = tempR[1];
            temp2_vy [threadIdx.y][threadIdx.x] = tempR[2];
            temp2_p  [threadIdx.y][threadIdx.x] = tempR[3];
            
        }
        __syncthreads();
        if(threadIdx.x < BDIMX_X - 3 && iglobal < nx + 1 && jglobal < ny + 4){
            int index_XL = threadIdx.x + 1;
            int index_XR = threadIdx.x;
            int index_Y  = threadIdx.y;

            double consL[4], consR[4];
            consL[0] = temp3_rho[index_Y][index_XL];
            consL[1] = temp3_vx [index_Y][index_XL];
            consL[2] = temp3_vy [index_Y][index_XL];
            consL[3] = temp3_p  [index_Y][index_XL];

            consR[0] = temp2_rho[index_Y][index_XR];
            consR[1] = temp2_vx [index_Y][index_XR];
            consR[2] = temp2_vy [index_Y][index_XR];
            consR[3] = temp2_p  [index_Y][index_XR];

            // ---------------- Step 1: 转换为原始量，并计算 x 方向通量 ----------------
            double priL[4], priR[4];
            get_pri(consL, priL);
            get_pri(consR, priR);

            double fluxL[4], fluxR[4];
            get_flux_x(priL, fluxL);
            get_flux_x(priR, fluxR);

            // ---------------- Step 2: 计算 LF 与 RI_U ----------------
            double LF[4], RI_U[4];
            for (int k = 0; k < 4; k++) {
                LF[k]   = 0.5 * (fluxL[k] + fluxR[k]) + 0.5 * (dx / dt) * (consR[k] - consL[k]);
                RI_U[k] = 0.5 * (consL[k] + consR[k]) - 0.5 * (dt / dx) * (fluxL[k] - fluxR[k]);
            }
            // ---------------- Step 3: 计算 RI 通量 ----------------
            double pri_RI[4], RI[4];
            get_pri(RI_U, pri_RI);
            get_flux_x(pri_RI, RI);
            // ---------------- Step 4: 计算最终 SLIC flux = 0.5*(LF + RI) ----------------
            double slic_flux[4];
            for (int k = 0; k < 4; k++) {
                slic_flux[k] = 0.5 * (LF[k] + RI[k]);
            }
            temp2_rho[threadIdx.y][threadIdx.x] = slic_flux[0];
            temp2_vx [threadIdx.y][threadIdx.x] = slic_flux[1];
            temp2_vy [threadIdx.y][threadIdx.x] = slic_flux[2];
            temp2_p  [threadIdx.y][threadIdx.x] = slic_flux[3];
        }
        __syncthreads();
        // start to update the data
        if (threadIdx.x < BDIMX_X - 4 && iglobal < nx && jglobal < ny + 4) {
            temp1_rho[threadIdx.y][threadIdx.x+2] = temp1_rho[threadIdx.y][threadIdx.x+2] - (dt/dx) * (temp2_rho[threadIdx.y][threadIdx.x + 1] - temp2_rho[threadIdx.y][threadIdx.x]);
            temp1_vx[threadIdx.y][threadIdx.x+2]  = temp1_vx[threadIdx.y][threadIdx.x+2]  - (dt/dx) * (temp2_vx [threadIdx.y][threadIdx.x + 1] - temp2_vx [threadIdx.y][threadIdx.x]);
            temp1_vy[threadIdx.y][threadIdx.x+2]  = temp1_vy[threadIdx.y][threadIdx.x+2]  - (dt/dx) * (temp2_vy [threadIdx.y][threadIdx.x + 1] - temp2_vy [threadIdx.y][threadIdx.x]);
            temp1_p[threadIdx.y][threadIdx.x+2]   = temp1_p[threadIdx.y][threadIdx.x+2]   - (dt/dx) * (temp2_p  [threadIdx.y][threadIdx.x + 1] - temp2_p  [threadIdx.y][threadIdx.x]);
        }


        __syncthreads();
        if (threadIdx.y < BDIMY_Y - 2  && threadIdx.x < BDIMX_X && iglobal < nx + 4 && jglobal < ny + 2) {
            int tempy = threadIdx.y + 1;
            double conM[4];  // con(i,j)
            double conL[4];  // con(i-1,j)
            double conR[4];  // con(i+1,j)
            conM[0] = temp1_rho[tempy][threadIdx.x];
            conM[1] = temp1_vx [tempy][threadIdx.x];  // 这里 vx 里实际存的是 rho*u
            conM[2] = temp1_vy [tempy][threadIdx.x];  // 这里 vy 里实际存的是 rho*v
            conM[3] = temp1_p  [tempy][threadIdx.x];  // E (总能量)

            conL[0] = temp1_rho[tempy - 1][threadIdx.x];
            conL[1] = temp1_vx [tempy - 1][threadIdx.x];
            conL[2] = temp1_vy [tempy - 1][threadIdx.x];
            conL[3] = temp1_p  [tempy - 1][threadIdx.x];

            conR[0] = temp1_rho[tempy + 1][threadIdx.x];
            conR[1] = temp1_vx [tempy + 1][threadIdx.x];
            conR[2] = temp1_vy [tempy + 1][threadIdx.x];
            conR[3] = temp1_p  [tempy + 1][threadIdx.x];
            // --- Step 2: 斜率限制，得到 tempL, tempR (仍在保守量空间) ---
            double tempL[4], tempR[4];
            for (int k = 0; k < 4; k++) {
                double temp1 = conM[k] - conL[k];  // i - (i-1)
                double temp2 = conR[k] - conM[k];  // (i+1) - i
                double di = 0.5 * (temp1 + temp2);

                // 这里分别调用 limiterL2 / limiterR2：
                double phiL = limiterL2(temp1, temp2);
                double phiR = limiterR2(temp1, temp2);

                // 得到左右临时状态
                tempL[k] = conM[k] - 0.5 * di * phiL;
                tempR[k] = conM[k] + 0.5 * di * phiR;
            }
            // --- Step 3: 将 tempL, tempR 转为原始量 priL, priR，并计算通量 fluxL, fluxR ---
            double priL[4], priR[4];
            get_pri(tempL, priL);
            get_pri(tempR, priR);

            double fluxL[4], fluxR[4];
            get_flux_y(priL, fluxL);
            get_flux_y(priR, fluxR);

            // --- Step 4: 半步更新 (回到保守量空间) ---
            // tempL, tempR 各减去 0.5*(dt/dx)*(fluxR - fluxL)
            for (int k = 0; k < 4; k++) {
                double delta = 0.5 * (dt / dy) * (fluxR[k] - fluxL[k]);
                tempL[k] = tempL[k] - delta;
                tempR[k] = tempR[k] - delta;
            }
            temp3_rho[threadIdx.y][threadIdx.x] = tempL[0];
            temp3_vx [threadIdx.y][threadIdx.x] = tempL[1];
            temp3_vy [threadIdx.y][threadIdx.x] = tempL[2];
            temp3_p  [threadIdx.y][threadIdx.x] = tempL[3];

            temp2_rho[threadIdx.y][threadIdx.x] = tempR[0];
            temp2_vx [threadIdx.y][threadIdx.x] = tempR[1];
            temp2_vy [threadIdx.y][threadIdx.x] = tempR[2];
            temp2_p  [threadIdx.y][threadIdx.x] = tempR[3];
            
        }
        __syncthreads();
        if(threadIdx.y < BDIMY_Y - 3 && iglobal < nx + 4 && jglobal < ny + 1){
            int index_YL = threadIdx.y + 1;
            int index_YR = threadIdx.y;
            int index_X  = threadIdx.x;

            double consL[4], consR[4];
            consL[0] = temp3_rho[index_YL][index_X];
            consL[1] = temp3_vx[index_YL][index_X];
            consL[2] = temp3_vy[index_YL][index_X];
            consL[3] = temp3_p[index_YL][index_X];

            consR[0] = temp2_rho[index_YR][index_X];
            consR[1] = temp2_vx[index_YR][index_X];
            consR[2] = temp2_vy[index_YR][index_X];
            consR[3] = temp2_p [index_YR][index_X];

            // ---------------- Step 1: 转换为原始量，并计算 x 方向通量 ----------------
            double priL[4], priR[4];
            get_pri(consL, priL);
            get_pri(consR, priR);

            double fluxL[4], fluxR[4];
            get_flux_y(priL, fluxL);
            get_flux_y(priR, fluxR);

            // ---------------- Step 2: 计算 LF 与 RI_U ----------------
            double LF[4], RI_U[4];
            for (int k = 0; k < 4; k++) {
                LF[k]   = 0.5 * (fluxL[k] + fluxR[k]) + 0.5 * (dy / dt) * (consR[k] - consL[k]);
                RI_U[k] = 0.5 * (consL[k] + consR[k]) - 0.5 * (dt / dy) * (fluxL[k] - fluxR[k]);
            }
            // ---------------- Step 3: 计算 RI 通量 ----------------
            double pri_RI[4], RI[4];
            get_pri(RI_U, pri_RI);
            get_flux_y(pri_RI, RI);
            // ---------------- Step 4: 计算最终 SLIC flux = 0.5*(LF + RI) ----------------
            double slic_flux[4];
            for (int k = 0; k < 4; k++) {
                slic_flux[k] = 0.5 * (LF[k] + RI[k]);
            }
            temp2_rho[threadIdx.y][threadIdx.x] = slic_flux[0];
            temp2_vx[threadIdx.y][threadIdx.x] = slic_flux[1];
            temp2_vy[threadIdx.y][threadIdx.x] = slic_flux[2];
            temp2_p[threadIdx.y][threadIdx.x] = slic_flux[3];
        }
        __syncthreads();
        // start to update the data
        if (threadIdx.y < BDIMY_Y - 4   && iglobal < nx && jglobal < ny) {
            temp1_rho[threadIdx.y+2][threadIdx.x] = temp1_rho[threadIdx.y+2][threadIdx.x]  - (dt/dy) * (temp2_rho[threadIdx.y+1][threadIdx.x] - temp2_rho[threadIdx.y][threadIdx.x]);
            temp1_vx [threadIdx.y+2][threadIdx.x] = temp1_vx [threadIdx.y+2][threadIdx.x]  - (dt/dy) * (temp2_vx [threadIdx.y+1][threadIdx.x] - temp2_vx [threadIdx.y][threadIdx.x]);
            temp1_vy [threadIdx.y+2][threadIdx.x] = temp1_vy [threadIdx.y+2][threadIdx.x]  - (dt/dy) * (temp2_vy [threadIdx.y+1][threadIdx.x] - temp2_vy [threadIdx.y][threadIdx.x]);
            temp1_p  [threadIdx.y+2][threadIdx.x]  = temp1_p  [threadIdx.y+2][threadIdx.x] - (dt/dy) * (temp2_p  [threadIdx.y+1][threadIdx.x] - temp2_p  [threadIdx.y][threadIdx.x]);
        }
        __syncthreads();
        if (threadIdx.y < BDIMY_Y - 4 && threadIdx.x < BDIMX_X - 4 && iglobal < nx && jglobal < ny){
            int stride_old = nx + 4;
            int idx = (jglobal+2) * stride_old + iglobal+2;
            d_data_con.rho[idx] = temp1_rho[threadIdx.y+2][threadIdx.x+2];
            d_data_con.vx[idx]  = temp1_vx [threadIdx.y+2][threadIdx.x+2];
            d_data_con.vy[idx]  = temp1_vy [threadIdx.y+2][threadIdx.x+2];
            d_data_con.p[idx]   = temp1_p  [threadIdx.y+2][threadIdx.x+2];
        }
        __syncthreads();
    }


    void launchUpdateSLICKernel(solVectors &d_data_con, double dt)
    {
        

        // dim3 block(BDIMX_X, BDIMY_Y);
        // dim3 grid(SHARE_X_GRID_X,SHARE_Y_GRID_Y);
        // compute_shared<<<grid, block>>>(d_data_con, dt, dx, nx, ny);
        // cudaDeviceSynchronize();
        dim3 block(BDIMX_X, BDIMX_Y);
        dim3 grid(SHARE_X_GRID_X,SHARE_X_GRID_Y);
        compute_x_shared<<<grid, block>>>(d_data_con, dt, dx, nx, ny);
        // cudaDeviceSynchronize();
        // cudaError_t err2 = cudaGetLastError();
        // if (err2 != cudaSuccess) {
        //     std::cerr << "CUDA kernel launch failed in share X s: " 
        //               << cudaGetErrorString(err2) << std::endl;
        //     exit(-1);
        // }
        // d_data_con = d_data_con;
        // cudaDeviceSynchronize();
        dim3 blocky(BDIMY_X, BDIMY_Y);
        dim3 gridy(SHARE_Y_GRID_X, SHARE_Y_GRID_Y);
        compute_y_shared<<<gridy, blocky>>>(d_data_con, dt, dy, nx, ny);
        // cudaDeviceSynchronize();
        // cudaError_t err = cudaGetLastError();
        // if (err != cudaSuccess) {
        //     std::cerr << "CUDA kernel launch failed in share Y: " 
        //               << cudaGetErrorString(err) << std::endl;
        //     exit(-1);
        // }
    }


    void checkKernelAttributes() {
    cudaFuncAttributes attr;

    cudaFuncGetAttributes(&attr, compute_shared);
    std::cout << "=== compute_shared Kernel Attributes ===" << std::endl;
    std::cout << "Registers used: " << attr.numRegs << std::endl;
    std::cout << "Shared memory per block: " << attr.sharedSizeBytes << " bytes" << std::endl;
    std::cout << "Constant memory used: " << attr.constSizeBytes << " bytes" << std::endl;
    std::cout << "Local memory per thread: " << attr.localSizeBytes << " bytes" << std::endl;
    std::cout << "Max threads per block: " << attr.maxThreadsPerBlock << std::endl;
    std::cout << std::endl;

    cudaFuncGetAttributes(&attr, compute_y_shared);
    std::cout << "=== compute_y_shared Kernel Attributes ===" << std::endl;
    std::cout << "Registers used: " << attr.numRegs << std::endl;
    std::cout << "Shared memory per block: " << attr.sharedSizeBytes << " bytes" << std::endl;
    std::cout << "Constant memory used: " << attr.constSizeBytes << " bytes" << std::endl;
    std::cout << "Local memory per thread: " << attr.localSizeBytes << " bytes" << std::endl;
    std::cout << "Max threads per block: " << attr.maxThreadsPerBlock << std::endl;
    std::cout << std::endl;
}