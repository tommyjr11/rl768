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

void compute_halftime_ULR_GPU(
    const double *d_old_u, // GPU 上的 old_u 数组
    double *d_half_uL,
    double *d_half_uR,
    int nx, int ny,
    double dt, double dx, double dy,
    int choice,
    double w,
    double r
);




#endif