#include "cuda.h"
#include "timer.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#include "kernel.h"



std::vector<int> cuda_histogram(const std::vector<float>& v, int num_bins, util::timer_pool& timers) {
    if (v.empty()) {
        std::cerr << "Error: Input vector is empty." << std::endl;
        return {};
    }

    float *d_v = nullptr;
    int *d_hist = nullptr;
    int vector_size = static_cast<int>(v.size());

    float min_val = *std::min_element(v.begin(), v.end());
    float max_val = *std::max_element(v.begin(), v.end());

    std::cout << "CUDA Min: " << min_val << ", Max: " << max_val << std::endl;

    cudaMalloc(&d_v, vector_size * sizeof(float));
    cudaMalloc(&d_hist, num_bins * sizeof(int));

    cudaMemset(d_hist, 0, num_bins * sizeof(int));

    cudaMemcpy(d_v, v.data(), vector_size * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (vector_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto& cuda_timer = timers.gpu_add("CUDA Histogram Computation", v.size());
    cuda_timer.do_start();

    histogram_kernel<<<blocks, BLOCK_SIZE>>>(d_v, d_hist, num_bins, vector_size, min_val, max_val);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
        return {};
    }

    cudaDeviceSynchronize();

    cuda_timer.do_stop();

    std::vector<int> hist_host(num_bins);
    cudaMemcpy(hist_host.data(), d_hist, num_bins * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_v);
    cudaFree(d_hist);

    return hist_host;
}
