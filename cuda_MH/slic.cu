#include <iostream>
#include <math.h>
#include <stdint.h>
#include "slic.h"

__device__ float limiterL(float smaller,float larger) {
    float R_slope = 0.0;
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
        float temp2 = 2*R_slope/(1+R_slope);
        return fminf(1.0, temp2);
        }  
}

__device__ float limiterL1(float smaller, float larger) {
    // 利用 __double_as_longlong 获取 larger 的位表示，
    // 然后取绝对值部分（屏蔽符号位），判断是否为 0
    double larger_d = static_cast<double>(larger);
    unsigned long long l_bits = __double_as_longlong(larger_d);
    unsigned long long l_abs  = l_bits & 0x7FFFFFFFFFFFFFFFULL;
    // 若 larger 非 0，则 (l_abs != 0ULL) 为 true 转换为 double 得 1.0，再取 1.0 - 1.0 = 0
    // 若 larger 为 0，则 (l_abs != 0ULL) 为 false 得 0，再取 1.0 - 0 = 1.0
    float isLargerZero = 1.0 - (float)(l_abs != 0ULL);

    // 同理，检测 smaller 是否为 0
    double smaller_d = static_cast<double>(smaller);
    unsigned long long s_bits = __double_as_longlong(smaller_d);
    unsigned long long s_abs  = s_bits & 0x7FFFFFFFFFFFFFFFULL;
    float isSmallerZero = 1.0 - (float)(s_abs != 0ULL);

    // 当 larger 非 0 时，直接计算 R = smaller / larger；
    // 当 larger 为 0 时，此除法结果可能为 NaN（0/0）或 ±∞（非 0/0），但后面会被掩盖
    float R = smaller / larger;
    // 对 R 限制在 [0, 1] 内（对于 R <= 0 返回 0，对于 0 < R <= 1 返回 R，对于 R > 1 返回 1）
    float clamped = fminf(fmaxf(R, 0.0), 1.0);

    // 当 larger 非 0 时，取 clamped；当 larger 为 0 时，根据原逻辑：
    //   - 若 smaller 为 0，返回 0；若 smaller 非 0，返回 1
    float isLargerNonZero = 1.0 - isLargerZero;
    float result_when_larger_zero = 1.0 - isSmallerZero;

    // 利用掩码实现条件选择：两部分相加
    return isLargerNonZero * clamped + isLargerZero * result_when_larger_zero;
}

__device__ float limiterR(float smaller,float larger) {
    float R_slope = 0.0;
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
        float temp2 = 2/(1+R_slope);
        return fminf(1.0, temp2);
    }
}

__device__ float limiterR1(float smaller, float larger) {
    // 利用 __double_as_longlong 及位掩码判断 larger 是否为 0
    double larger_d = static_cast<double>(larger);
    unsigned long long l_bits = __double_as_longlong(larger_d);
    unsigned long long l_abs  = l_bits & 0x7FFFFFFFFFFFFFFFULL;
    // isLargerZero 为 1.0 当 larger==0，否则为 0.0
    float isLargerZero = 1.0 - (float)(l_abs != 0ULL);
    float isLargerNonZero = 1.0 - isLargerZero;
    
    // 避免除零：若 larger==0，则 safeLarger = larger + 1 = 1，从而计算 R 时乘以 isLargerNonZero 得 0
    float safeLarger = larger + isLargerZero;
    float R = (smaller / safeLarger) * isLargerNonZero;
    
    // 构造条件掩码
    float cond_R_gt_0 = (R > 0.0);      // 当 R>0 时为 1.0，否则 0.0
    float cond_R_le_1 = (R <= 1.0);      // 当 R<=1 时为 1.0，否则 0.0
    float cond_R_gt_1 = 1.0 - cond_R_le_1; // R>1 时为 1.0
    
    // 根据 R 的取值范围选择返回值：
    // - 当 0 < R <= 1 时，返回 R；
    // - 当 R > 1 时，返回 2/(1+R)；
    // - 当 R <= 0 时，返回 0（由 cond_R_gt_0 掩码控制）。
    float part1 = cond_R_le_1 * R;
    float part2 = cond_R_gt_1 * (2.0 / (1.0 + R));
    float result_inner = part1 + part2;
    
    return cond_R_gt_0 * result_inner;
}





