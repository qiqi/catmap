#pragma once
#include<cinttypes>

__device__ __forceinline__
void pIncrement(
        float * tmpCounts,
        double dx, double dy, uint32_t nx, uint32_t ny,
        double xy[4]
) {
    if (xy[0] < 0 or xy[1] < 0)
        return;
    uint32_t i = uint32_t(xy[0] / dx);
    uint32_t j = uint32_t(xy[1] / dy);
    
    if (i < nx and j < ny) {
        uint32_t ind00 = i * ny + j;
        uint32_t ind01 = i * ny + (j + 1) % ny;
        uint32_t ind10 = (i + 1) % nx * ny + j;
        uint32_t ind11 = (i + 1) % nx * ny + (j + 1) % ny;

        double fracx = fmod(xy[0], dx);
        double fracy = fmod(xy[1], dy);

        atomicAdd(tmpCounts + ind00, -1.0f);
        atomicAdd(tmpCounts + ind01, +1.0f);
    }
}

template<bool accumulate, class Map>
__global__
void pRun(
        Map map, uint32_t nIters,
        uint32_t nPoints, double (*points)[4],
        float * tmpCounts,
        double dx, double dy, uint32_t nx, uint32_t ny
) {
    uint32_t ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < nPoints) {
        double ptx[4] = {points[ind][0],
                         points[ind][1],
                         points[ind][2],
                         points[ind][3]};
        for (uint32_t iIter = 0; iIter < nIters; ++iIter) {
            map.map(ptx);
            if (accumulate) {
                pIncrement(tmpCounts, dx, dy, nx, ny, ptx);
            }
        }
        points[ind][0] = ptx[0];
        points[ind][1] = ptx[1];
        points[ind][2] = ptx[2];
        points[ind][3] = ptx[3];
    }
}

__global__
void pAddToBigCounterAndClear(
        double * bigCounts, float * tmpCounts, uint32_t n
) {
    uint32_t ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < n) {
        bigCounts[ind] += tmpCounts[ind];
        tmpCounts[ind] = 0;
    }
}


struct BilinearCounter {
  public:
    const uint32_t nx, ny;
    const double dx, dy;
    double* counts;

  private:
    float* tmpCounts;
    double* bigCounts;
    double* cpuCounts;

    double (*points)[4];
    uint32_t nPoints;

    void pAddToCpuCounterAndClear() {
        cudaMemcpy(cpuCounts, bigCounts,
                   sizeof(double) * nx * ny, cudaMemcpyDeviceToHost);
        cudaMemset(bigCounts, 0, sizeof(double) * nx * nx);
        for (uint32_t i = 0; i < nx * ny; ++i) {
            counts[i] += cpuCounts[i];
        }
    }

  public:
    BilinearCounter(uint32_t nx, uint32_t ny, double dx, double dy)
        : nx(nx), ny(ny), dx(dx), dy(dy) {
        cudaMalloc(&tmpCounts, sizeof(float) * nx * nx);
        cudaMalloc(&bigCounts, sizeof(double) * nx * nx);
        cpuCounts = new double[nx * nx];
        counts = new double[nx * nx];
        memset(counts, 0, sizeof(double) * nx * nx);
        nPoints = 0;
    }

    void init(uint32_t nPoints) {
        cudaMemset(tmpCounts, 0, sizeof(float) * nx * nx);
        cudaMemset(bigCounts, 0, sizeof(double) * nx * nx);

        if (this->nPoints) {
            cudaFree(points);
        }
        this->nPoints = nPoints;
        cudaMalloc(&points, sizeof(double) * nPoints * 4);

        double *cpuPoints = new double[nPoints*4];
        for (uint32_t i = 0; i < nPoints; ++i) {
            cpuPoints[i * 4 + 0] = (rand() / double(RAND_MAX) + rand()) / double(RAND_MAX);
            cpuPoints[i * 4 + 1] = (rand() / double(RAND_MAX) + rand()) / double(RAND_MAX);
            cpuPoints[i * 4 + 2] = 1.0;
            cpuPoints[i * 4 + 3] = 0.0;
        }
        cudaMemcpy(points, cpuPoints, sizeof(double) * nPoints * 4,
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

    ~BilinearCounter() {
        cudaFree(tmpCounts);
        cudaFree(bigCounts);
        delete[] cpuCounts;
        delete[] counts;
    }
};
