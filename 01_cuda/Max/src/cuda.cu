#include "cuda.h"
#include "timer.h"
#include <iostream>
#include <vector>
#include "kernel.h"  
#include <algorithm>
#include <cuda_runtime.h>


#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

float cuda_max_value(const std::vector<float>& v, util::timer_pool& timers) {
    if (v.empty()) {
        std::cerr << "Error: Input vector is empty." << std::endl;
        return -1;
    }

    float *d_v = nullptr, *d_max = nullptr;
    int vector_size = static_cast<int>(v.size());
    int blocks = (vector_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    
    cudaMalloc(&d_v, vector_size * sizeof(float));
    cudaMalloc(&d_max, blocks * sizeof(float));

    
    cudaMemcpy(d_v, v.data(), vector_size * sizeof(float), cudaMemcpyHostToDevice);

    
    auto& cuda_timer = timers.gpu_add("CUDA Max Value Computation", v.size());
    cuda_timer.do_start();

    
    max_kernel<<<blocks, BLOCK_SIZE>>>(d_v, vector_size, d_max);

    
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    
    cudaDeviceSynchronize();

    
    cuda_timer.do_stop();

    
    std::vector<float> max_host(blocks);
    cudaMemcpy(max_host.data(), d_max, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    
    float max_value = max_host[0];
    for (float val : max_host) {
        max_value = std::max(max_value, val);
    }

    
    cudaFree(d_v);
    cudaFree(d_max);

    return max_value;
}
