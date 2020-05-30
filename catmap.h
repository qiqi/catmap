#pragma once
#include<cmath>

struct CatMap {
    const float s0, s1, s2, s3;

    __device__ __forceinline__
    void map(float x[2]) {
        const float scale = 2 * M_PI * (5*5 + 8*8);
        float y[2] = {2 * x[0] + x[1], x[0] + x[1]};
    
        float d0 = sin(2 * M_PI * (8 * x[0] + 5 * x[1])) / scale;
        float d1 = sin(2 * M_PI * (5 * x[0] - 8 * x[1])) / scale;
    
        y[0] += s0 * d1 * 5;
        y[1] -= s0 * d1 * 8;
    
        y[0] += s1 * d1 * 8;
        y[1] += s1 * d1 * 5;
    
        y[0] += s2 * d0 * 5;
        y[1] -= s2 * d0 * 8;
    
        y[0] += s3 * d0 * 8;
        y[1] += s3 * d0 * 5;
    
        x[0] = fmodf(y[0] + 1, 1.0);
        x[1] = fmodf(y[1] + 1, 1.0);
    }
};


