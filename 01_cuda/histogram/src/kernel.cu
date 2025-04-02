#include "kernel.h"
#include "timer.h"
#include "gen.h"

#include <bits/stdc++.h>
#include <stdio.h>

typedef std::mt19937 RNG;  // Mersenne Twister with a popular choice of parameters

using namespace std;

#define BLOCK_SIZE 256
#define NUM_READS 32


// Define a kernel function, which is the entry point
// for execution on the GPU
__global__ void kernel(float *v_g, uint32_t *h, const float l, const float r, const int n, const uint64_t vn) {
    int bi = blockIdx.x;
    uint64_t i  = blockIdx.x*blockDim.x*NUM_READS + threadIdx.x;
    int ti = threadIdx.x;

    extern __shared__ uint32_t h_s[];

    if (ti == 0) {
        for (int k=0; k<n ; k++) {
            h_s[k] = 0;
        }
    }

    __syncthreads();

    for (int k=0; k<NUM_READS; k++) {
        if (i+k*BLOCK_SIZE < vn) {
            int hi = (v_g[i+k*BLOCK_SIZE] - l) / (r - l) * n;

            hi = max(hi, 0);
            hi = min(hi, n-1);

            atomicAdd(&h_s[hi], 1);
        }
    }

    __syncthreads();

    if (ti == 0) {
        for (int k=0; k<n; k++) {
            atomicAdd(&h[k], h_s[k]);
        }
    }
}


// Define a wrapper function, which launches the kernel
void kernel_wrapper(vector<float> &v, const float l, const float r, const int n) {
    vector<uint32_t> h(n);

    float *v_g;

    uint32_t *h_g;

    cudaMalloc(&v_g, v.size()*sizeof(v[0]));
    cudaMalloc(&h_g, n*sizeof(h[0]));


    cudaMemcpy(v_g, v.data(), v.size()*sizeof(v[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(h_g, h.data(), h.size()*sizeof(h[0]), cudaMemcpyHostToDevice);

    auto& timer = util::timers.gpu_add("max");

    dim3 grid(ceil((float)v.size()/(BLOCK_SIZE*NUM_READS)), 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);

    kernel<<<grid, block, n*sizeof(h[0])>>>(v_g, h_g, l, r, n, v.size());

    //while (n > 1) {

    //    dim3 grid(ceil((float)n/(BLOCK_SIZE*NUM_READS)), 1, 1);
    //    dim3 block(BLOCK_SIZE, 1, 1);

    //    // Launch kernel with <<<block, thread>>> syntax
    //    kernel<Max><<<grid, block>>>(b, a, n);

    //    swap(a, b);
    //    n = ceil((float)n/(BLOCK_SIZE*NUM_READS));
    //}
    //swap(a, b);
    timer.stop();

    cudaMemcpy(h.data(), h_g, h.size()*sizeof(h[0]), cudaMemcpyDeviceToHost);

    cout << endl;

    for (int k=0; k<h.size(); k++) {
        cout << h[k] << endl;
    }

    //cout << "GPU max : " << r[0] << endl;

}


// Define a wrapper function, which launches the kernel
void cpu_only(vector<float> &v) {
    auto& timer = util::timers.cpu_add("max");

    float m = *max_element(v.begin(), v.end());
    timer.stop();

    cout << "CPU max : " << m << endl;

}
