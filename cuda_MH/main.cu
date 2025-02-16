#include <iostream>
#include "constants.h"
#include "data.h"

int main() {
    // int   limiter_CPU = 2;
    // float gamma_CPU[3];
    // gamma_CPU[0] = 1.4;

    // gamma_CPU[1] = (gamma_CPU[0] - 1.f) / (2.f * gamma_CPU[0]);
    // gamma_CPU[2] = (gamma_CPU[0] + 1.f) / (2.f * gamma_CPU[0]);

    // cudaMemcpyToSymbol(limiter, &limiter_CPU, sizeof(limiter_CPU), 0, cudaMemcpyHostToDevice);
    // cudaMemcpyToSymbol(g, &gamma_CPU, sizeof(gamma_CPU), 0, cudaMemcpyHostToDevice);

    solVectors d_data;
    allocateDeviceMemory(d_data);
    initDataAndCopyToGPU(d_data);

    std::cout << "test" << std::endl;
    float dt = getdtGPU(d_data, 1.4f);
    std::cout << "dt = " << dt << std::endl;

    freeDeviceMemory(d_data);
    return 0;
}