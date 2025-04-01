#include <vector>
#include "kernel.h"

#include <stdio.h>
#include <iostream>

#include <cuda/std/complex>
#include <random>

typedef std::mt19937 RNG;  // Mersenne Twister with a popular choice of parameters

using namespace cuda::std;

// Define a device function, which 
// can be called from a kernel and executes on the GPU
__device__ int device_function(complex<double> c){
    complex<double> z(0.0,0.0);

    int i;
    for (i = 0; i < 10; i++){
        z = z*z + c;
        if (abs(z) > 2.0) return i;
    }

    return i;
}

// Define a kernel function, which is the entry point
// for execution on the GPU
__global__ void kernel() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = threadIdx.x;

    complex<double> c(i, ti);
    auto r = device_function(c);
}

// Define a wrapper function, which launches the kernel

void kernel_wrapper() {
    // Launch kernel with <<<block, thread>>> syntax
    kernel<<<10,32>>>();
    const complex<double> i(0.0,1.0);    
    std::cout << "Complex test: "<< i.real() << "," << i.imag() << std::endl;
}
