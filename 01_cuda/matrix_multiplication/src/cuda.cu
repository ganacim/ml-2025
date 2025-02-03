#include "cuda.h"
#include "timer.h"

#include <stdio.h>
#include <iostream>
#include <exception>
#include <cmath>

using namespace std;

// Define the block size. It is a good idea to use a power of 2
// for the block size to maximize memory coalescing and avoid bank conflicts.
// Size is the width and height of the block. The total number of threads
// in a block is BLOCK_SIZE * BLOCK_SIZE.
const unsigned int BLOCK_SIZE = 16;

// Define a kernel function, which is the entry point
// for execution on the GPU
__global__ void matrix_multiplication(float *m1, float *m2, float *result, unsigned int m1_rows, unsigned int m1_cols, unsigned int m2_cols)
{
    // Get the row and column of the current element
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

    // Return if the current element is out of bounds
    if (i >= m1_rows || j >= m2_cols) {
        return;
    }

    // Compute the dot product of the row of m1 and the column of m2
    float value = 0;
    for (unsigned int k = 0; k < m1_cols; k++) {
        value += m1[i * m1_cols + k] * m2[k * m2_cols + j];
    }

    // Store the result in the output matrix
    result[i * m2_cols + j] = value;
}

vector<float> cuda_multiplication(const std::vector<float>& m1,
                                    const std::vector<float>& m2,
                                    unsigned int m1_rows,
                                    unsigned int m1_cols,
                                    unsigned int m2_cols)
{
    auto& timer = util::timers.gpu_add("CUDA Multiplication");
    // Allocate memory on the host
    vector<float> result(m1_rows * m2_cols);
    // Allocate memory on the device
    float *d_m1, *d_m2, *d_result;
    cudaMalloc(&d_m1, m1_rows * m1_cols * sizeof(float));
    cudaMalloc(&d_m2, m1_cols * m2_cols * sizeof(float));
    cudaMalloc(&d_result, m1_rows * m2_cols * sizeof(float));
    // Copy data from host to device
    cudaMemcpy(d_m1, m1.data(), m1_rows * m1_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, m2.data(), m1_cols * m2_cols * sizeof(float), cudaMemcpyHostToDevice);
    // // sync cuda device
    // cudaDeviceSynchronize();
    // Define grid and block size
    dim3 grid(ceil((float)m1_rows/BLOCK_SIZE), ceil((float)m2_cols/BLOCK_SIZE), 1);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    // cout << "grid: " << grid.x << " " << grid.y << " " << grid.z << endl;
    // cout << "block: " << block.x << " " << block.y << " " << block.z << endl;
    // Launch kernel
    matrix_multiplication<<<grid, block>>>(d_m1, d_m2, d_result, m1_rows, m1_cols, m2_cols);
    // // sync cuda device
    // cudaDeviceSynchronize();
    // Copy data from device to host
    cudaMemcpy(result.data(), d_result, m1_rows * m2_cols * sizeof(float), cudaMemcpyDeviceToHost);
    // Free memory on the device
    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_result);
    timer.stop();
    return result;
}


__global__ void matrix_block_multiplication(float *m1, float *m2, float *result, unsigned int m1_rows, unsigned int m1_cols, unsigned int m2_cols)
{
    // Get the row and column of the current element
    unsigned int ti = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + ti;
    unsigned int tj = threadIdx.y;
    unsigned int j = blockIdx.y*blockDim.y + tj;

    __shared__ float m1_s[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float m2_s[BLOCK_SIZE][BLOCK_SIZE];

    float r = 0.f;
    for (unsigned int k=0; k < m1_cols/BLOCK_SIZE; k++) {
        m1_s[ti][tj] = m1[i*m1_cols + k*BLOCK_SIZE + tj];
        m2_s[ti][tj] = m2[(k*BLOCK_SIZE + ti)*m2_cols + j];
        __syncthreads();

        for (unsigned int l=0; l<BLOCK_SIZE; ++l) {
            r += m1_s[ti][l]*m2_s[l][tj];
        }
        __syncthreads();
    }
    result[i * m2_cols + j] = r;
}

vector<float> cuda_block_multiplication(const std::vector<float>& m1,
                                    const std::vector<float>& m2,
                                    unsigned int m1_rows,
                                    unsigned int m1_cols,
                                    unsigned int m2_cols)
{
    auto& timer = util::timers.gpu_add("CUDA Block Multiplication");
    // Allocate memory on the host
    vector<float> result(m1_rows * m2_cols);
    // Allocate memory on the device
    float *d_m1, *d_m2, *d_result;
    cudaMalloc(&d_m1, m1_rows * m1_cols * sizeof(float));
    cudaMalloc(&d_m2, m1_cols * m2_cols * sizeof(float));
    cudaMalloc(&d_result, m1_rows * m2_cols * sizeof(float));
    // Copy data from host to device
    cudaMemcpy(d_m1, m1.data(), m1_rows * m1_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, m2.data(), m1_cols * m2_cols * sizeof(float), cudaMemcpyHostToDevice);
    // // sync cuda device
    // cudaDeviceSynchronize();
    // Define grid and block size
    dim3 grid(ceil((float)m1_rows/BLOCK_SIZE), ceil((float)m2_cols/BLOCK_SIZE), 1);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    // cout << "grid: " << grid.x << " " << grid.y << " " << grid.z << endl;
    // cout << "block: " << block.x << " " << block.y << " " << block.z << endl;
    // Launch kernel
    matrix_block_multiplication<<<grid, block>>>(d_m1, d_m2, d_result, m1_rows, m1_cols, m2_cols);
    // // sync cuda device
    // cudaDeviceSynchronize();
    // Copy data from device to host
    cudaMemcpy(result.data(), d_result, m1_rows * m2_cols * sizeof(float), cudaMemcpyDeviceToHost);
    // Free memory on the device
    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_result);
    timer.stop();
    return result;
}