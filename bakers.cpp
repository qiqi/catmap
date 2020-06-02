#include<cstdio>
#include<cassert>
#include<algorithm>

#include"bakersmap.h"
#include"density.h"

int main() {
    uint32_t iDevice;
    assert(fread(&iDevice, sizeof(uint32_t), 1, stdin) == 1);
    cudaSetDevice(iDevice);
    fprintf(stderr, "Set To Device %u\n", iDevice);

    uint32_t randSeed;
    assert(fread(&randSeed, sizeof(uint32_t), 1, stdin) == 1);
    srand(randSeed);

    const int nx = 2048;
    Counter counter(nx, nx, 2 * M_PI / nx, 2 * M_PI / nx);

    float parameters[4];
    assert(fread(parameters, sizeof(float), 4, stdin) == 4);
    BakersMap map{parameters[0], parameters[1], parameters[2], parameters[3]};
    fprintf(stderr, "Ready with parameters %f %f %f %f\n",
            parameters[0], parameters[1], parameters[2], parameters[3]);

    uint32_t nIters;
    assert(fread(&nIters, sizeof(uint32_t), 1, stdin) == 1);
    for (uint32_t iIter = 0; iIter < nIters; iIter+=4) {
        counter.init(80000);
        counter.run(map, 1, 256, false);
        if (iIter % 128 == 0) {
            fprintf(stderr, "%u/%u iterations\n", iIter, nIters);
        }
        counter.run(map, std::min(4u, nIters - iIter), 1024, true);
    }

    fwrite(counter.counts, sizeof(double), nx * nx, stdout);
    return 0;
}
