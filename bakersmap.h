#pragma once
#include<cmath>

struct BakersMap {
    const double s0, s1, s2, s3;

    __device__ __forceinline__
    void map(double x[2]) {
        double y[2] = {2 * x[0], (x[1] + floor(x[0] / M_PI) * 2 * M_PI) / 2.};
    
        double d0 = sin(2 * x[0]) / 2.0;
        double d1 = sin(x[1]);
    
        y[1] += s0 * d1;
        y[0] += s1 * d1 * d0;
        y[1] += s2 * d1 * d0;
        y[0] += s3 * d0;
    
        x[0] = fmod(y[0], 2 * M_PI);
        x[1] = fmod(y[1], 2 * M_PI);
    }
};


