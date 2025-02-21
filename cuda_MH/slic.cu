#include <iostream>
#include <math.h>
#include <stdint.h>
#include "slic.h"

__device__ double limiterL(double smaller,double larger) {
    double R_slope = 0.0;
    if (smaller == 0 && larger == 0){
        R_slope = 0.0;
    }
    else if (larger == 0 && smaller != 0){
        return 1.0;
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
        double temp2 = 2*R_slope/(1+R_slope);
        return fmin(1.0, temp2);
        }  
}

__device__ double limiterL1(double smaller, double larger) {
    // 利用 __double_as_longlong 获取 larger 的位表示，
    // 然后取绝对值部分（屏蔽符号位），判断是否为 0
    unsigned long long l_bits = __double_as_longlong(larger);
    unsigned long long l_abs  = l_bits & 0x7FFFFFFFFFFFFFFFULL;
    // 若 larger 非 0，则 (l_abs != 0ULL) 为 true 转换为 double 得 1.0，再取 1.0 - 1.0 = 0
    // 若 larger 为 0，则 (l_abs != 0ULL) 为 false 得 0，再取 1.0 - 0 = 1.0
    double isLargerZero = 1.0 - (double)(l_abs != 0ULL);

    // 同理，检测 smaller 是否为 0
    unsigned long long s_bits = __double_as_longlong(smaller);
    unsigned long long s_abs  = s_bits & 0x7FFFFFFFFFFFFFFFULL;
    double isSmallerZero = 1.0 - (double)(s_abs != 0ULL);

    // 当 larger 非 0 时，直接计算 R = smaller / larger；
    // 当 larger 为 0 时，此除法结果可能为 NaN（0/0）或 ±∞（非 0/0），但后面会被掩盖
    double R = smaller / larger;
    // 对 R 限制在 [0, 1] 内（对于 R <= 0 返回 0，对于 0 < R <= 1 返回 R，对于 R > 1 返回 1）
    double clamped = fmin(fmax(R, 0.0), 1.0);

    // 当 larger 非 0 时，取 clamped；当 larger 为 0 时，根据原逻辑：
    //   - 若 smaller 为 0，返回 0；若 smaller 非 0，返回 1
    double isLargerNonZero = 1.0 - isLargerZero;
    double result_when_larger_zero = 1.0 - isSmallerZero;

    // 利用掩码实现条件选择：两部分相加
    return isLargerNonZero * clamped + isLargerZero * result_when_larger_zero;
}

__device__ double limiterL2(double smaller, double larger) {
    if (larger == 0.0)
        return (smaller == 0.0) ? 0.0 : 1.0;
    double R = smaller / larger;
    return fmin(fmax(R, 0.0), 1.0);
}

__device__ double limiterR(double smaller,double larger) {
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
        return fmin(1.0, temp2);
    }
}

__device__ double limiterR1(double smaller, double larger) {
    // 利用 __double_as_longlong 及位掩码判断 larger 是否为 0
    unsigned long long l_bits = __double_as_longlong(larger);
    unsigned long long l_abs  = l_bits & 0x7FFFFFFFFFFFFFFFULL;
    // isLargerZero 为 1.0 当 larger==0，否则为 0.0
    double isLargerZero = 1.0 - (double)(l_abs != 0ULL);
    double isLargerNonZero = 1.0 - isLargerZero;
    
    // 避免除零：若 larger==0，则 safeLarger = larger + 1 = 1，从而计算 R 时乘以 isLargerNonZero 得 0
    double safeLarger = larger + isLargerZero;
    double R = (smaller / safeLarger) * isLargerNonZero;
    
    // 构造条件掩码
    double cond_R_gt_0 = (R > 0.0);      // 当 R>0 时为 1.0，否则 0.0
    double cond_R_le_1 = (R <= 1.0);      // 当 R<=1 时为 1.0，否则 0.0
    double cond_R_gt_1 = 1.0 - cond_R_le_1; // R>1 时为 1.0
    
    // 根据 R 的取值范围选择返回值：
    // - 当 0 < R <= 1 时，返回 R；
    // - 当 R > 1 时，返回 2/(1+R)；
    // - 当 R <= 0 时，返回 0（由 cond_R_gt_0 掩码控制）。
    double part1 = cond_R_le_1 * R;
    double part2 = cond_R_gt_1 * (2.0 / (1.0 + R));
    double result_inner = part1 + part2;
    
    return cond_R_gt_0 * result_inner;
}

__device__ double limiterR2(double smaller, double larger) {
    if (larger == 0.0)
        return 0.0;
    double R = smaller / larger;
    return (R <= 0.0) ? 0.0 : ((R <= 1.0) ? R : fmin(1.0, 2.0/(1.0+R)));
}

