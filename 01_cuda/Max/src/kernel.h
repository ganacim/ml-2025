#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>

__global__ void max_kernel(float *d_v, int n, float *d_max);

#endif 
