#include "kernel.h"

#include <stdio.h>

// Define a device function, which 
// can be called from a kernel and executes on the GPU
__device__ int media(){
    printf("Hello CUDA World!\n");
    return 1;
}

// Define a kernel function, which is the entry point
// for execution on the GPU
__global__ void kernel() {
    media();
}

// Define a wrapper function, which launches the kernel
void kernel_wrapper() {
    // Launch kernel with <<<block, thread>>> syntax
    kernel<<<1,32>>>();
}
