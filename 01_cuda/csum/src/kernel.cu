#include "kernel.h"
#include <vector>
#include <stdio.h>

using namespace std;

#define BLOCK_SIZE 32

__global__ void kernel() {

}

// Define a wrapper function, which launches the kernel
void kernel_wrapper() {
    vector<float> v(32);
    for (int i=0; i<32; i++) {
        v[i] = i;
    }

    float* x;
    cudaMalloc(&x, v.size()*sizeof(v[0]));

    cudaMemcpy(x, v.data(), v.size()*sizeof(v[0]), cudaMemcpyHostToDevice);

    // Launch kernel with <<<block, thread>>> syntax
    kernel<<<1,32>>>();
}
