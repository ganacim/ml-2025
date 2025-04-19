#ifndef KERNEL_H
#define KERNEL_H

__global__ void histogram_kernel(const float* d_v, int* d_hist, int num_bins, int vector_size, float min_val, float max_val);

#endif
