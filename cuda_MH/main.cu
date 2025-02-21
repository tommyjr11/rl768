#include <iostream>
#include "constants.h"
#include "data.h"

int main() {
    solVectors d_data;
    allocateDeviceMemory(d_data);
    initDataAndCopyToGPU(d_data);
    float dt = getdtGPU(d_data, 1.4f);
    std::cout << "dt = " << dt << std::endl;
    float t = 0.0f;
    for (;;){
        if (t >= t1) break;
        t = t + dt;
        applyBoundaryConditions(d_data);
        // 继续实现
    }

    freeDeviceMemory(d_data);
    return 0;
}