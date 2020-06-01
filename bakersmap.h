#pragma once
#include<cmath>

struct BakersMap {
    const double s0, s1, s2, s3;

    __device__ __forceinline__
    void map(double x[2]) {
        const double scale = 2 * M_PI;
        double y[2] = {fmodf(2 * x[0], 1), (x[1] + floor(2 * x[0])) / 2.};
    
        double d0 = sin(2 * M_PI * x[0]) / scale;
        double d1 = sin(2 * M_PI * x[1]) / scale;
    
        y[1] += s0 * d1;
        y[0] += s1 * d1;
        y[1] -= s2 * d0;
        y[0] += s3 * d0;
    
        x[0] = y[0];
        x[1] = y[1];
    }
};


