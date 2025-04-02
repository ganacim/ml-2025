#include "kernel.h"
#include <vector>

#include <iostream>
#include <stdio.h>

#include <ctime>
#include <random>

typedef std::mt19937
    RNG; // Mersenne Twister with a popular choice of parameters
using namespace std;

#define BLOCK_SIZE 2

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

void kernel_wrapper() {
  vector<float> v = create_random_vector(1e9);
  // vector<float> v = {-2.0, -3.0, -1.0, -4.0};

  float max_val = v[0];
  for (int i = 1; i < v.size(); i++) {
    max_val = max(max_val, v[i]);
  }
  cout << "Cpu max value: " << max_val << endl;
}
