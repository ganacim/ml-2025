#include "kernel.h"
#include <cuda_runtime.h>

__global__ void histogram_kernel(const float* d_v, int* d_hist, int num_bins, int vector_size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < vector_size) {
        float range = max_val - min_val;
        int bin = min(num_bins - 1, max(0, static_cast<int>(((d_v[idx] - min_val) / range) * num_bins)));

        atomicAdd(&d_hist[bin], 1);
    }
}
