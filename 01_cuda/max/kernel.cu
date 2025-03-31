#include <vector>
#include "kernel.h"

#include <stdio.h>
#include <iostream>

#include <chrono>
#include <algorithm>
#include <ctime>
#include <random>

typedef std::mt19937 RNG;  // Mersenne Twister with a popular choice of parameters

using namespace std;

#define BLOCK_SIZE 32

__global__ void max_kernel(float *d_v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = threadIdx.x;
    __shared__ float sdata[BLOCK_SIZE];

    sdata[ti] = (i < n) ? d_v[i] : -__FLT_MAX__;
    __syncwarp();
    
    for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2){
        if (ti < stride){
            sdata[ti] = max(sdata[ti], sdata[ti + stride]);
        }
        __syncwarp();
    }

    if (ti == 0){
        d_v[blockIdx.x] = sdata[0];
    }
}

__global__ void min_kernel(float *d_v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = threadIdx.x;
    __shared__ float sdata[BLOCK_SIZE];

    sdata[ti] = (i < n) ? d_v[i] : __FLT_MAX__;
    __syncwarp();
    
    for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2){
        if (ti < stride){
            sdata[ti] = min(sdata[ti], sdata[ti + stride]);
        }
        __syncwarp();
    }

    if (ti == 0){
        d_v[blockIdx.x] = sdata[0];
    }
}

__global__ void sum_kernel(float *d_v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = threadIdx.x;
    __shared__ float sdata[BLOCK_SIZE];

    sdata[ti] = (i < n) ? d_v[i] : 0;
    __syncwarp();
    
    for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2){
        if (ti < stride){
            sdata[ti] += sdata[ti + stride];
        }
        __syncwarp();
    }

    if (ti == 0){
        d_v[blockIdx.x] = sdata[0];
    }
}


vector<float> create_random_vector(unsigned int cols)
{
    // Create a normal distribution with mean 0 and standard deviation 1
    uint32_t seed = (uint32_t) time(0);    
    RNG rng(seed);
    normal_distribution<float> normal(0.0, 1.0);
    // Create a matrix of size matrix_size x matrix_size with random values
    vector<float> matrix(cols);
    for (int j = 0; j < cols; j++) {
        matrix[j] = normal(rng);
    }
    return matrix;
}

double max_vector(const vector<float>& vec){
    double max_val = *max_element(vec.begin(), vec.end());
    return max_val;
}

double min_vector(const vector<float>& vec){
    double min_val = *min_element(vec.begin(), vec.end());
    return min_val;
}

double sum_vector(const vector<float>& vec){
    double sum = 0;
    for (auto v: vec){
        sum += v;
    }
    return sum;
}

void kernel_wrapper() {
    vector<float> v = create_random_vector(1 << 20);

    auto start_cpu = chrono::high_resolution_clock::now();
    double max_val = max_vector(v);
    double min_val = min_vector(v);
    double sum_val = sum_vector(v);
    auto end_cpu = chrono::high_resolution_clock::now();
    auto delta_cpu = chrono::duration<double, milli>(end_cpu - start_cpu).count();

    cout << "v.size(): " << v.size() << endl;
    cout << "True Max value: " << max_val << endl; 
    cout << "True Min value: " << min_val << endl; 
    cout << "True Sum value: " << sum_val << endl; 
    cout << "CPU elapsed time (ms): " << delta_cpu << endl;
    
    auto start_gpu = chrono::high_resolution_clock::now();

    int num_elements = v.size();
    int num_blocks = (num_elements - 1) / BLOCK_SIZE + 1;

    float *d_max;
    cudaMalloc(&d_max, num_elements * sizeof(float));
    cudaMemcpy(d_max, v.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);

    float *d_min;
    cudaMalloc(&d_min, num_elements * sizeof(float));
    cudaMemcpy(d_min, v.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);

    float *d_sum;
    cudaMalloc(&d_sum, num_elements * sizeof(float));
    cudaMemcpy(d_sum, v.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);

    while (num_elements > 1){
        dim3 grid(num_blocks);
        dim3 block(BLOCK_SIZE);

        max_kernel<<<grid, block>>>(d_max, num_elements);
        min_kernel<<<grid, block>>>(d_min, num_elements);
        sum_kernel<<<grid, block>>>(d_sum, num_elements);
        
        num_elements = num_blocks;
        num_blocks = (num_elements - 1) / BLOCK_SIZE + 1;
    }

    auto end_gpu = chrono::high_resolution_clock::now();
    auto delta_gpu = chrono::duration<double, milli>(end_gpu - start_gpu).count();

    vector<float> val_host(num_elements);
    cudaMemcpy(val_host.data(), d_max, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    max_val = max_vector(val_host);

    cudaMemcpy(val_host.data(), d_min, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    min_val = min_vector(val_host);

    cudaMemcpy(val_host.data(), d_sum, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    sum_val = sum_vector(val_host);

    cout << "GPU Max Value: " << max_val << endl;
    cout << "GPU Min Value: " << min_val << endl;
    cout << "GPU Sum Value: " << sum_val << endl;
    cout << "GPU elapsed time (ms): " << delta_gpu << endl;

    cudaFree(d_max);
    cudaFree(d_min);
    cudaFree(d_sum);
}
