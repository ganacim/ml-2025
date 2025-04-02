#include "cuda.h"
#include "timer.h"
#include <stdio.h>
#include <iostream>
#include <exception>
#include <cmath>
#include <vector>

using namespace std;

// Define a kernel function, which is the entry point
// for execution on the GPU


#define BLOCK_SIZE 32

__global__ void matrix_conv_wkernel(const float *m_in, 
                            const float *kernel, 
                            float *result, 
                            unsigned int m_rows, 
                            unsigned int m_cols,
                            unsigned int kernel_size)
{
    // Get the row and column of the current element
    unsigned int ti = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + ti;
    unsigned int tj = threadIdx.y;
    unsigned int j = blockIdx.y*blockDim.y + tj;

    if (i < m_cols-kernel_size+1 && j < m_rows-kernel_size+1){
        //int k_center = kernel_size / 2;
        float value = 0.0f;

        for (unsigned int x=0; x < kernel_size; x++) {
            for (unsigned int y=0; y < kernel_size; y++) { 
                int mrow = j + y;
                int mcol = i + x;

                if (mrow < m_rows && mcol < m_cols) {
                    value += m_in[mrow * m_cols + mcol] * kernel[x * kernel_size + y];
                }
            }
            
        }
    if (j < m_rows-kernel_size+1 && i < m_cols-kernel_size+1){
        result[j * (m_cols-kernel_size+1) + i] = value;
    }
    }
}

vector<float> cuda_convolution_template(const std::vector<float>& m,
                                const std::vector<float>& k,
                                unsigned int rows,
                                unsigned int cols,
                                unsigned int kernel_size)
{
    string name = "CUDA Conv2d";
    auto& timer = util::timers.gpu_add(name);

    // Allocate memory on the host
    int out_cols = cols - kernel_size  + 1;
    int out_rows = rows - kernel_size + 1;
    vector<float> result(out_rows * out_cols);

    // Allocate memory on the device
    float *d_m, *d_kern, *d_result;
    cudaMalloc(&d_m, rows * cols * sizeof(float));
    cudaMalloc(&d_kern, kernel_size * kernel_size * sizeof(float));
    cudaMalloc(&d_result, out_rows * out_cols * sizeof(float));
    // Copy data from host to device
    cudaMemcpy(d_m, m.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kern, k.data(), kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    // // sync cuda device
    // cudaDeviceSynchronize();
    // Define grid and block size
    int grid_size_y = (out_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_size_x = (out_cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(grid_size_x, grid_size_y);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    matrix_conv_wkernel<<<grid, block>>>(d_m, d_kern, d_result, rows, cols, kernel_size);
    
    // Launch kernel
    // // sync cuda device
    // cudaDeviceSynchronize();
    // Copy data from device to host
    cudaMemcpy(result.data(), d_result, out_rows * out_cols * sizeof(float), cudaMemcpyDeviceToHost);
    // Free memory on the device
    cudaFree(d_m);
    cudaFree(d_kern);
    cudaFree(d_result);
    timer.stop();
    return result;
}