#pragma once
#include<cinttypes>

__device__ __forceinline__
void pIncrement(
        uint32_t * tmpCounts,
        double dx, double dy, uint32_t nx, uint32_t ny,
        double xy[2]
) {
    // const double sina = 0.85065081, cosa = 0.52573111;
    const double sina = 0., cosa = 1.;
    double rotx = (xy[0] * sina + xy[1] * cosa - 0.5 * (sina + cosa)) + 0.5;
    double roty = (xy[0] * cosa - xy[1] * sina - 0.5 * (cosa - sina)) + 0.5;
    if (rotx < 0 or roty < 0)
        return;
    uint32_t i = uint32_t(rotx / dx);
    uint32_t j = uint32_t(roty / dy);
    
    if (i < nx and j < ny) {
        uint32_t ind = i * ny + j;
        atomicAdd(tmpCounts + ind, 1);
    }
}

template<bool accumulate, class Map>
__global__
void pRun(
        Map map, uint32_t nIters,
        uint32_t nPoints, double (*points)[2],
        uint32_t * tmpCounts,
        double dx, double dy, uint32_t nx, uint32_t ny
) {
    uint32_t ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < nPoints) {
        double ptx[2] = {points[ind][0], points[ind][1]};
        for (uint32_t iIter = 0; iIter < nIters; ++iIter) {
            map.map(ptx);
            if (accumulate) {
                pIncrement(tmpCounts, dx, dy, nx, ny, ptx);
            }
        }
        points[ind][0] = ptx[0];
        points[ind][1] = ptx[1];
    }
}

__global__
void pAddToBigCounterAndClear(
        uint64_t * bigCounts, uint32_t * tmpCounts, uint32_t n
) {
    uint32_t ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < n) {
        bigCounts[ind] += tmpCounts[ind];
        tmpCounts[ind] = 0;
    }
}


struct Counter {
  public:
    const uint32_t nx, ny;
    const double dx, dy;
    double* counts;

  private:
    uint32_t* tmpCounts;
    uint64_t* bigCounts;
    uint64_t* cpuCounts;

    double (*points)[2];
    uint32_t nPoints;

    void pAddToCpuCounterAndClear() {
        cudaMemcpy(cpuCounts, bigCounts, sizeof(uint64_t) * nx * ny,
                   cudaMemcpyDeviceToHost);
        cudaMemset(bigCounts, 0, sizeof(uint64_t) * nx * nx);
        for (uint32_t i = 0; i < nx * ny; ++i) {
            counts[i] += (double)cpuCounts[i];
        }
    }

  public:
    Counter(uint32_t nx, uint32_t ny, double dx, double dy)
        : nx(nx), ny(ny), dx(dx), dy(dy) {
        cudaMalloc(&tmpCounts, sizeof(uint32_t) * nx * nx);
        cudaMalloc(&bigCounts, sizeof(uint64_t) * nx * nx);
        cpuCounts = new uint64_t[nx * nx];
        counts = new double[nx * nx];
        memset(counts, 0, sizeof(double) * nx * nx);
        nPoints = 0;
    }

    void init(uint32_t nPoints) {
        cudaMemset(tmpCounts, 0, sizeof(uint32_t) * nx * nx);
        cudaMemset(bigCounts, 0, sizeof(uint64_t) * nx * nx);

        if (this->nPoints) {
            cudaFree(points);
        }
        this->nPoints = nPoints;
        cudaMalloc(&points, sizeof(double) * nPoints * 2);

        double *cpuPoints = new double[nPoints*2];
        for (uint32_t i = 0; i < nPoints * 2; ++i) {
            cpuPoints[i] = (rand() / double(RAND_MAX) + rand()) / double(RAND_MAX);
        }
        cudaMemcpy(points, cpuPoints, sizeof(double) * nPoints * 2,
                   cudaMemcpyHostToDevice);
        delete[] cpuPoints;
    }

    template<class Map>
    void run(Map map, uint32_t nBatches, uint32_t iterPerBatch, bool accumulate) {
        if (accumulate) {
            for (uint32_t iBatch = 0; iBatch < nBatches; ++iBatch) {
                pRun<true, Map><<<ceil(nPoints / 64.), 64>>>(
                        map, iterPerBatch, nPoints, points,
                        tmpCounts, dx, dy, nx, ny);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("After pRun<true>, CudaError = %d\n", err);
                    exit(-1);
                }
                pAddToBigCounterAndClear<<<ceil(nx * ny / 64.), 64>>>(
                        bigCounts, tmpCounts, nx * ny);
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("After pAddToBigCounterAndClear, CudaError = %d\n", err);
                    exit(-1);
                }
            }
            pAddToCpuCounterAndClear();
        } else {
            for (uint32_t iBatch = 0; iBatch < nBatches; ++iBatch) {
                pRun<false, Map><<<ceil(nPoints / 64.), 64>>>(
                        map, iterPerBatch, nPoints, points,
                        tmpCounts, dx, dy, nx, ny);
                cudaDeviceSynchronize();
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("After pRun<false>, CudaError = %d\n", err);
                    exit(-1);
                }
            }
        }
        cudaDeviceSynchronize();
    }

    ~Counter() {
        cudaFree(tmpCounts);
        cudaFree(bigCounts);
        delete[] cpuCounts;
        delete[] counts;
    }
};
