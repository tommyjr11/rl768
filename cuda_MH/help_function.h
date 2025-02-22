#ifndef HELP_FUNCTION_H
#define HELP_FUNCTION_H
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




#endif