#include "kernel.h"
#include <cfloat>

#define BLOCK_SIZE 64

__global__ void max_kernel(float *d_v, int n, float *d_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = threadIdx.x;
    __shared__ float sdata[BLOCK_SIZE];

    if (i < n) {
        sdata[ti] = d_v[i];
    } else {
        sdata[ti] = -FLT_MAX;
    }
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (ti < stride) {
            sdata[ti] = max(sdata[ti], sdata[ti + stride]);
        }
        __syncthreads();
    }

    if (ti == 0) {
        d_max[blockIdx.x] = sdata[0];
    }
}
