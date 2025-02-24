#include <iostream>
#include "data.h"
#include <vector>
#include <chrono>
int main() {
    auto start = std::chrono::high_resolution_clock::now();
    solVectors d_data_pri;
    solVectors d_data_con;
    solVectors d_half_uL;
    solVectors d_half_uR;
    solVectors d_SLIC_flux;
    allocateDeviceMemory(d_data_pri, d_data_con);
    initDataAndCopyToGPU(d_data_pri, d_data_con);
    double dt = 0.0f;
    double t = 0.0f;
    int step = 0;
    for (;;){
        dt = getdtGPU(d_data_pri, 1.4f);
        // std::cout << "step: "<< step << " dt = " << dt << std::endl;
        step++;
        if (t >= t1) break;
        t = t + dt;
        applyBoundaryConditions(d_data_con);
        // x 方向
        computeHalftime(d_data_con,d_half_uL,d_half_uR,dt,1);
        computeSLICFlux(d_half_uL,d_half_uR,d_SLIC_flux,dt,1);
        updateSolution(d_data_con,d_SLIC_flux,dt,1);
        freeDeviceMemory2(d_half_uL, d_half_uR, d_SLIC_flux);

        computeHalftime(d_data_con,d_half_uL,d_half_uR,dt,2);
        computeSLICFlux(d_half_uL,d_half_uR,d_SLIC_flux,dt,2);
        updateSolution(d_data_con,d_SLIC_flux,dt,2);
        freeDeviceMemory2(d_half_uL, d_half_uR, d_SLIC_flux);
        list_con2pri(d_data_con, d_data_pri);
    }
    std::vector<double> h_rho((nx+4) * (ny+4), 0.0f);
    std::vector<double> h_vx((nx+4) * (ny+4), 0.0f);
    std::vector<double> h_vy((nx+4) * (ny+4), 0.0f);
    std::vector<double> h_p((nx+4) * (ny+4), 0.0f);
    cudaMemcpy(h_rho.data(), d_data_pri.rho, sizeof(double) * (nx+4) * (ny+4), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vx.data(), d_data_pri.vx, sizeof(double) * (nx+4) * (ny+4), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vy.data(), d_data_pri.vy, sizeof(double) * (nx+4) * (ny+4), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_p.data(), d_data_pri.p, sizeof(double) * (nx+4) * (ny+4), cudaMemcpyDeviceToHost);
    store_data(h_rho, h_vx, h_vy, h_p,dt,1);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time: " << elapsed.count() << " s\n";
    freeDeviceMemory(d_data_pri, d_data_con);
    return 0;
}