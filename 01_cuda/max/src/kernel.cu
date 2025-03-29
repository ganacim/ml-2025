#include "kernel.h"
#include <vector>

#include <iostream>
#include <stdio.h>

#include <ctime>
#include <random>

typedef std::mt19937
    RNG; // Mersenne Twister with a popular choice of parameters
using namespace std;

#define BLOCK_SIZE 3

__global__ void max_kernel(float *d_v, float *d_max) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int ti = threadIdx.x;
  __shared__ float sdata[BLOCK_SIZE];

  sdata[ti] = d_v[i];
  __syncthreads();

  float max_val = sdata[0];
  if (ti == 0) {
    for (int k = 1; k < BLOCK_SIZE; k++) {
      max_val = max(max_val, sdata[k]);
    }
  }
  d_max[blockIdx.x] = max_val;
}

vector<float> create_random_vector(unsigned int cols) {
  // Create a normal distribution with mean 0 and standard deviation 1
  uint32_t seed = (uint32_t)time(0);
  RNG rng(seed);
  normal_distribution<float> normal(0.0, 1.0);
  // Create a matrix of size matrix_size x matrix_size with random values
  vector<float> matrix(cols);
  for (int j = 0; j < cols; j++) {
    matrix[j] = normal(rng);
  }
  return matrix;
}

// FIXME: Not working when BLOCK_SIZE doesn't divide the size of the vector

void kernel_wrapper() {
  vector<float> v = create_random_vector(4);

  cout << "Original v: ";
  for (auto i : v) {
    cout << i << " ";
  }
  cout << endl;

  float *d_v;
  cudaMalloc(&d_v, v.size() * sizeof(float));
  cudaMemcpy(d_v, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);

  int length_max_vector;

  // NOTE: For a, b integers: (a + (b - 1)) / b = ceil(a/b)
  length_max_vector = (v.size() + (BLOCK_SIZE - 1)) / BLOCK_SIZE;

  while (length_max_vector > 1) {
    float *d_max;
    cudaMalloc(&d_max, length_max_vector * sizeof(float));

    dim3 grid(length_max_vector);
    dim3 block(BLOCK_SIZE);

    max_kernel<<<grid, block>>>(d_v, d_max);

    d_v = d_max;
    length_max_vector = (length_max_vector + (BLOCK_SIZE - 1)) / BLOCK_SIZE;
  }
  float final_max;
  cudaMemcpy(&final_max, d_v, sizeof(float), cudaMemcpyDeviceToHost);
  cout << "Final max value: " << final_max << endl;

  // Real max value
  float max_val = v[0];
  for (int i = 1; i < v.size(); i++) {
    max_val = max(max_val, v[i]);
  }
  cout << "Real max value: " << max_val << endl;

  cudaFree(d_v);
}
