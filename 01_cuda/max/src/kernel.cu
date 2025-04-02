#include <vector>
#include "kernel.h"
#include "timer.h"
#include <stdio.h>
#include <iostream>
#include <random>


typedef std::mt19937 RNG;  // Mersenne Twister with a popular choice of parameters

using namespace std;

#define BLOCK_SIZE 32
//Como feito em aula
__global__ void max_kernel(float *d_v, int n, float *d_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = threadIdx.x;
    __shared__ float sdata[BLOCK_SIZE];

    if (i < n) {
        sdata[ti] = d_v[i];  
    } else {
        sdata[ti] = -INFINITY;  
    }
    __syncthreads();

    float max_val = sdata[0];
    if (ti == 0){
        for (int k = 1; k < BLOCK_SIZE; k++){
            max_val = max(max_val, sdata[k]);
        }
    
    d_max[blockIdx.x] = max_val;}
}

void kernel_wrapper(vector<float> v) {

    string name = "CUDA Vector Max value";
    

    cout << "v.size(): " << v.size() << endl;

    float *d_v;
    cudaMalloc(&d_v, v.size() * sizeof(float));
    cudaMemcpy(d_v, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);

    float *d_max;

    int grid_size = (v.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc(&d_max, grid_size * sizeof(float));

    auto& timer = util::timers.gpu_add(name);
    dim3 block(BLOCK_SIZE);
    int size = v.size();

    //Vetor é diminuído por cada bloco
    //Isso é repetido até que reste um valor, 
    //que é o máximo
    while (size > 1){

        grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 grid(grid_size);

        max_kernel<<<grid, block>>>(d_v, size, d_max);

        size = grid_size;

        if (size > 1){
            float* temp = d_v;   
            d_v = d_max;         
            d_max = temp;        
        }

    }
    timer.stop();
    vector<float> max_val_host(size);
    cudaMemcpy(max_val_host.data(), d_max, size * sizeof(float), cudaMemcpyDeviceToHost);
    cout << endl;
    cudaFree(d_v);
    cudaFree(d_max);

}
