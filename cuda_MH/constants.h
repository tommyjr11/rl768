// constants.h

#ifndef CONSTANTS_H
#define CONSTANTS_H

// 声明 GPU 常量
extern __constant__ int   limiter;
extern __constant__ float g[9];

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

#endif // CONSTANTS_H
