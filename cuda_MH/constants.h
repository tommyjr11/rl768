#ifndef CONSTANTS_H
#define CONSTANTS_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef DEFINE_CONSTANTS
    // 当 DEFINE_CONSTANTS 被定义时，直接定义，不加 extern
    #define CONSTEXTERN 
#else
    // 否则，只做声明
    #define CONSTEXTERN extern
#endif

CONSTEXTERN __constant__ int limiter;
CONSTEXTERN __constant__ float g[3];

#ifdef __cplusplus
}
#endif

// 一些可能用到的枚举
enum Vars {
    RHO,
    V_X,
    V_Y,
    P,
    NUM_VARS
};

enum Limiter {
    firstOrder,
    minBee,
    vanLeer,
    superBee
};

enum Processor {
    CPU,
    GPU
};

// 清除宏定义，防止污染全局
#undef CONSTEXTERN

#endif // CONSTANTS_H
