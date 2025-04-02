#include "kernel.h"
#include "timer.h"
#include "gen.h"

#include <bits/stdc++.h>
#include <stdio.h>

typedef std::mt19937 RNG;  // Mersenne Twister with a popular choice of parameters

using namespace std;

#define BLOCK_SIZE 128
#define NUM_READS 32

class Max {
public:
    __device__ float operator()(float a, float b) {
        return max(a, b);
    }
};

class Min {
public:
    __device__ float operator()(float a, float b) {
        return min(a, b);
    }
};

class Sum {
public:
    __device__ float operator()(float a, float b) {
        return a + b;
    }
};

// Define a kernel function, which is the entry point
// for execution on the GPU
template <class Func>
__global__ void kernel(float *b, float *a, const int n) {
    Func func;

    int bi = blockIdx.x;
    int i  = blockIdx.x*blockDim.x*NUM_READS + threadIdx.x;
    int ti = threadIdx.x;

    __shared__ float b_s[BLOCK_SIZE];

    if (i < n)
        b_s[ti] = b[i];

    for (int k=1; k<NUM_READS; k++) {
        if (i+k*BLOCK_SIZE < n)
            b_s[ti] = func(b_s[ti], b[i+k*BLOCK_SIZE]);
    }

    __syncthreads();

    if (ti == 0) {
        float m = b_s[0];

        int nj = min(BLOCK_SIZE, n-i);
        for (int j=1; j<nj; j++) {
            m = func(m, b_s[j]);
        }
        a[bi] = m;
    }
}


// Define a wrapper function, which launches the kernel
void kernel_wrapper(vector<float> &v) {

    int n = v.size();

    vector<float> r(ceil((float)n/(BLOCK_SIZE*NUM_READS)));

    float *b, *a;


    cudaMalloc(&b, n*sizeof(v[0]));
    cudaMalloc(&a, ceil((float)n/(BLOCK_SIZE*NUM_READS))*sizeof(v[0]));

    cudaMemcpy(b, v.data(), n*sizeof(v[0]), cudaMemcpyHostToDevice);

    auto& timer = util::timers.gpu_add("max");


    while (n > 1) {

        dim3 grid(ceil((float)n/(BLOCK_SIZE*NUM_READS)), 1, 1);
        dim3 block(BLOCK_SIZE, 1, 1);

        // Launch kernel with <<<block, thread>>> syntax
        kernel<Max><<<grid, block>>>(b, a, n);

        swap(a, b);
        n = ceil((float)n/(BLOCK_SIZE*NUM_READS));
    }
    swap(a, b);
    timer.stop();

    cudaMemcpy(r.data(), a, sizeof(v[0]), cudaMemcpyDeviceToHost);
    cout << "GPU max : " << r[0] << endl;

}


// Define a wrapper function, which launches the kernel
void cpu_only(vector<float> &v) {
    auto& timer = util::timers.cpu_add("max");

    float m = *max_element(v.begin(), v.end());
    timer.stop();

    cout << "CPU max : " << m << endl;

}
