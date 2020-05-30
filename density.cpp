#include<cstdio>
#include<cassert>

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
    counter.init(80000);
    counter.run(cat, 0, 256, false);
    fprintf(stderr, "Ready with parameters %f %f %f %f\n",
            parameters[0], parameters[1], parameters[2], parameters[3]);

    uint32_t nIters;
    assert(fread(&nIters, sizeof(uint32_t), 1, stdin) == 1);
    fprintf(stderr, "Running %u iterations\n", nIters);
    counter.run(cat, nIters, 1024, true);

    fwrite(counter.counts, sizeof(double), nx * nx, stdout);
    return 0;
}
