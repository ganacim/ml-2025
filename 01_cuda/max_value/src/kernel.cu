#include <vector>
#include "kernel.h"
#include <cfloat> 

#include <stdio.h>
#include <iostream>
#include <chrono>

#include <ctime>
#include <random>

using namespace std::chrono;

using namespace std;


#define BLOCK_SIZE 1024

__global__ void max_kernel(float *d_v, int n, float *d_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = threadIdx.x;
    __shared__ float sdata[BLOCK_SIZE];

    sdata[ti] = (i < n) ? d_v[i] : -FLT_MAX;
    __syncthreads();

    float max_val = sdata[0];
    if (ti == 0){
        for (int k = 1; k < BLOCK_SIZE; k++){
            max_val = max(max_val, sdata[k]);
        }
    }
    d_max[blockIdx.x] = max_val;
}


void kernel_wrapper(vector<float> v) {

    // Make sure that the data is a power of 32
    int top_power = 1;
    int iteration = 1;
    int input_size = v.size();
    while (top_power<input_size)
    {
        top_power *= BLOCK_SIZE;
        iteration++;
    }
    int div = v.size()/BLOCK_SIZE;
    

    float *d_v;
    cudaMalloc(&d_v, v.size() * sizeof(float));
    cudaMemcpy(d_v, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);

    float *d_max;
    cudaMalloc(&d_max, (v.size() /BLOCK_SIZE+1) * sizeof(float));

    cout << "Number of iterations: " << iteration << endl;
    auto start = high_resolution_clock::now();
    while(iteration>1){
        int grid_size = input_size/BLOCK_SIZE +1;
        dim3 grid(grid_size);
        dim3 block(BLOCK_SIZE);

        max_kernel<<<grid, block>>>(d_v, input_size, d_max);
        d_v = d_max;
        iteration--;
        input_size = grid_size;

    }
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    cout << "Execution Time in GPU: " << duration.count() << " us" << endl;

    vector<float> max_val_host(1);

    // Save the maximum list to a list in the cpu
    cudaMemcpy(max_val_host.data(), d_max, sizeof(float), cudaMemcpyDeviceToHost);

    // for (auto v: max_val_host){
    //     cout << v << " ";
    // }
    cout <<  "The maximum value in GPU was: " << max_val_host[0] << endl;
    // cout << endl;
    cudaFree(d_v);
}
