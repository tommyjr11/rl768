#include <iostream>
#include "constants.h"
#include "data.h"
#include <vector>

int main() {
    solVectors d_data_pri;
    solVectors d_data_con;
    solVectors d_half_uL;
    solVectors d_half_uR;
    allocateDeviceMemory(d_data_pri, d_data_con);
    initDataAndCopyToGPU(d_data_pri, d_data_con);
    float dt = getdtGPU(d_data_pri, 1.4f);
    std::vector<float> h_rho((nx+2) * (ny+4), 0.0f);
    // cudaMemcpy(h_rho.data(), d_data_con.vx, (nx+4) * (ny+4) * sizeof(float), cudaMemcpyDeviceToHost);
    // // 打印 rho
    // for (int j = 0; j < ny+4; j++){
    //     for (int i = 0; i < nx+4; i++){
    //         std::cout << h_rho[j * (nx+4) + i] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    std::cout << "dt = " << dt << std::endl;
    float t = 0.0f;
    for (;;){
        if (t >= t1) break;
        t = t + dt;
        applyBoundaryConditions(d_data_pri);
        // 继续实现
        
        computeHalftime(d_data_con,d_half_uL,d_half_uR,dt,1);
        cudaMemcpy(h_rho.data(), d_half_uL.rho, (nx+2) * (ny+4) * sizeof(float), cudaMemcpyDeviceToHost);
        // 打印 rho
        for (int j = 0; j < ny+4; j++){
            for (int i = 0; i < nx+2; i++){
                std::cout << h_rho[j * (nx+2) + i] << " ";
            }
            std::cout << std::endl;
        }
        exit(0);    
    }

    freeDeviceMemory(d_data_pri, d_data_con);
    return 0;
}