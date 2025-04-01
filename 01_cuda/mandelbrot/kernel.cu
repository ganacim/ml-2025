#include <vector>
#include "kernel.h"

#include <stdio.h>
#include <iostream>

#include <cuda/std/complex>
#include <random>

typedef std::mt19937 RNG;  // Mersenne Twister with a popular choice of parameters

using namespace cuda::std;

const int BLOCK_SIZE = 16;
// Define a kernel function, which is the entry point
// for execution on the GPU
__global__ void mandelbrot_kernel(int *d_res, const int WIDTH, const int HEIGHT, const float scale, const float cx, const float cy) {
    unsigned int ti = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + ti;
    unsigned int tj = threadIdx.y;
    unsigned int j = blockIdx.y*blockDim.y + tj;
    
    if (i >= WIDTH || j >= HEIGHT){
        return;
    }

    const int max_iter = 20;
    const double x = (i/WIDTH - 0.5) * scale + cx;
    const double y = (j/HEIGHT - 0.5) * scale + cy;

    complex<double> c(x, y), z(0, 0);
    __syncthreads();

    int k = 0;
    while(abs(z) < 2 && k < max_iter){
        z = z*z + c;
        k++;
    }

    d_res[i * HEIGHT + j] = k;
    __syncthreads();

    return;
}

// Define a wrapper function, which launches the kernel
void kernel_wrapper(int* result, const int WIDTH, const int HEIGHT, const float scale, const float cx, const float cy) {
    // Launch kernel with <<<block, thread>>> syntax
    int *d_result;
    cudaMalloc(&d_result, WIDTH * HEIGHT * sizeof(int));

    dim3 grid(ceil((float)WIDTH/BLOCK_SIZE), ceil((float)HEIGHT/BLOCK_SIZE), 1);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    
    mandelbrot_kernel<<<grid, block>>>(d_result, WIDTH, HEIGHT, scale, cx, cy);

    cudaMemcpy(result, d_result, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}
