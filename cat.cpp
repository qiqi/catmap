#include<cstdio>
#include<cassert>
#include<algorithm>

#include"catmap.h"
#include"density.h"

int main() {
    uint32_t iDevice;
    assert(fread(&iDevice, sizeof(uint32_t), 1, stdin) == 1);
    cudaSetDevice(iDevice);
    fprintf(stderr, "Set To Device %u\n", iDevice);

    const int nx = 2048;
    Counter counter(nx, nx, 1./nx, 1./nx);

    float parameters[4];
    assert(fread(parameters, sizeof(float), 4, stdin) == 4);
    CatMap cat{parameters[0], parameters[1], parameters[2], parameters[3]};
    fprintf(stderr, "Ready with parameters %f %f %f %f\n",
            parameters[0], parameters[1], parameters[2], parameters[3]);

    uint32_t nIters;
    assert(fread(&nIters, sizeof(uint32_t), 1, stdin) == 1);
    for (uint32_t iIter = 0; iIter < nIters; iIter) {
        counter.init(80000);
        counter.run(cat, 0, 256, false);
        if (iIter % 128 == 0) {
            fprintf(stderr, "%u/%u iterations\n", iIter, nIters);
        }
        counter.run(cat, 1, 1024, true);
    }

    fwrite(counter.counts, sizeof(double), nx * nx, stdout);
    return 0;
}
