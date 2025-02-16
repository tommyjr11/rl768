// constants.h (或 constants.cuh)

#ifndef CONSTANTS_H
#define CONSTANTS_H

// 声明外部符号
extern __constant__ int limiter;
extern __constant__ float g[9];

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
