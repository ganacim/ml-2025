#include "kernel.h"
#include <cmath>
#include <iostream>

using namespace std;

// Define a kernel function, which is the entry point
// for execution on the GPU
__global__ void mandelbrot(int *result) {}

// Define a wrapper function, which launches the kernel
void kernel_wrapper() {
  int height = 128;
  int width = 128;

  int block_size = 32;

  dim3 grid(height / block_size, width / block_size);
  dim3 block(block_size, block_size);

  int *result;
  cudaMalloc(&result, height * width * sizeof(int));

  // Launch kernel with <<<block, thread>>> syntax
  mandelbrot<<<grid, block>>>(result);
}
