#include <iostream>
#include "data.h"
#include <vector>
#include <chrono>
int main() {
    
    solVectors d_data_pri;
    solVectors d_data_con;
    // solVectors d_half_uL;
    // solVectors d_half_uR;
    // solVectors d_SLIC_flux;
    checkKernelAttributes();
    allocateDeviceMemory(d_data_pri, d_data_con);
    initDataAndCopyToGPU2(d_data_pri, d_data_con);
    auto start = std::chrono::high_resolution_clock::now();
    double dt = 0.0;
    double t = 0.0;
    int step = 0;
    double tempt = 0.0f;
    double temptt = bubbleR/(sqrt(r*pAir/rhoAir)*Ms);
    std::chrono::duration<double, std::milli> launch_duration;
    // std::cout<<"shack rho: "<< rhoPost<<"shack p: "<< pPost<<"shack u: "<< uPost<<"shack v: "<< vPost<<std::endl;
    for (;;){
        // auto dt_start = std::chrono::high_resolution_clock::now();
        dt = getdtGPU(d_data_pri, 1.4);
        // cudaDeviceSynchronize();
        // auto dt_end = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> dt_duration = dt_end - dt_start;
        // std::cout << "dt time: " << dt_duration.count() << " millisecond"<<std::endl;

        // cudaDeviceSynchronize();
        std::cout << "step: "<< step << " dt = " << dt <<" t= "<<t<<std::endl;
        step++;
        t = t + dt;
        // auto launch_start = std::chrono::high_resolution_clock::now();
        launchUpdateSLICKernel(d_data_con, dt);
        // auto launch_end = std::chrono::high_resolution_clock::now();
        // launch_duration = launch_duration + (launch_end - launch_start);
        // std::cout << "launch time: " << launch_duration.count() << " millisecond"<<std::endl;


    
        // cudaDeviceSynchronize();
        // auto x_half_start = std::chrono::high_resolution_clock::now();
        // computeHalftime(d_data_con,d_half_uL,d_half_uR,dt,1);
        // cudaDeviceSynchronize();
        // auto x_half_end = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> x_half_duration = x_half_end - x_half_start;
        // std::cout << "x_half time: " << x_half_duration.count() << " millisecond"<<std::endl;

        // auto x_slic_start = std::chrono::high_resolution_clock::now();
        // computeSLICFlux(d_half_uL,d_half_uR,d_SLIC_flux,dt,1);
        // cudaDeviceSynchronize();
        // auto x_slic_end = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> x_slic_duration = x_slic_end - x_slic_start;
        // std::cout << "x_slic time: " << x_slic_duration.count() << " millisecond"<<std::endl;

        // auto x_update_start = std::chrono::high_resolution_clock::now();
        // updateSolution(d_data_con,d_SLIC_flux,dt,1);
        // cudaDeviceSynchronize();
        // freeDeviceMemory2(d_half_uL, d_half_uR, d_SLIC_flux);
        // auto x_update_end = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> x_update_duration = x_update_end - x_update_start;
        // std::cout << "x_update time: " << x_update_duration.count() << " millisecond"<<std::endl;

        

        // auto y_half_start = std::chrono::high_resolution_clock::now();
        // computeHalftime(d_data_con,d_half_uL,d_half_uR,dt,2);
        // cudaDeviceSynchronize();
        // auto y_half_end = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> y_half_duration = y_half_end - y_half_start;
        // std::cout << "y_half time: " << y_half_duration.count() << " millisecond"<<std::endl;

        // auto y_slic_start = std::chrono::high_resolution_clock::now();
        // computeSLICFlux(d_half_uL,d_half_uR,d_SLIC_flux,dt,2);
        // cudaDeviceSynchronize();
        // auto y_slic_end = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> y_slic_duration = y_slic_end - y_slic_start;
        // std::cout << "y_slic time: " << y_slic_duration.count() << " millisecond"<<std::endl;

        // auto y_update_start = std::chrono::high_resolution_clock::now();
        // updateSolution(d_data_con,d_SLIC_flux,dt,2);
        // cudaDeviceSynchronize();
        // freeDeviceMemory2(d_half_uL, d_half_uR, d_SLIC_flux);
        // auto y_update_end = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> y_update_duration = y_update_end - y_update_start;
        // std::cout << "y_update time: " << y_update_duration.count() << " millisecond"<<std::endl;

        applyBoundaryConditions(d_data_con);
        // cudaDeviceSynchronize();
        list_con2pri(d_data_con, d_data_pri);
        // cudaDeviceSynchronize();
        tempt = t/temptt;
        if (tempt >= t1) break;
        // if (t >= t1) break;
    }
    // std::vector<double> h_rho((nx+4) * (ny+4), 0.0f);
    // std::vector<double> h_vx((nx+4) * (ny+4), 0.0f);
    // std::vector<double> h_vy((nx+4) * (ny+4), 0.0f);
    // std::vector<double> h_p((nx+4) * (ny+4), 0.0f);
    // cudaMemcpy(h_rho.data(), d_data_pri.rho, sizeof(double) * (nx+4) * (ny+4), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_vx.data(), d_data_pri.vx, sizeof(double) * (nx+4) * (ny+4), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_vy.data(), d_data_pri.vy, sizeof(double) * (nx+4) * (ny+4), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_p.data(), d_data_pri.p, sizeof(double) * (nx+4) * (ny+4), cudaMemcpyDeviceToHost);
    // store_data(h_rho, h_vx, h_vy, h_p,dt,1);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time: " << elapsed.count() << " s\n";
    freeDeviceMemory(d_data_pri, d_data_con);
    return 0;
}