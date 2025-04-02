#include "matrix.h"

#include <iostream>
#include <fstream>
#include <ctime>
#include <random>

typedef std::mt19937 RNG;  // Mersenne Twister with a popular choice of parameters

using namespace std;

vector<float> create_random_matrix(unsigned int rows, unsigned int cols)
{
    // Create a normal distribution with mean 0 and standard deviation 1
    uint32_t seed = (uint32_t) time(0);    
    RNG rng(seed);
    normal_distribution<float> normal(0.0, 1.0);
    // Create a matrix of size matrix_size x matrix_size with random values
    vector<float> matrix(rows * cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = normal(rng);
        }
    }
    return matrix;
}

vector<float> create_kernel(unsigned int k){
    // Create a matrix of size matrix_size x matrix_size with random values
    vector<float> kernel(k * k);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            kernel[i * k + j] = (float)1 / (k*k);
        }
    }
    return kernel;
}

void print_matrix(const vector<float> &matrix, unsigned int rows)
{
    unsigned int cols = matrix.size() / rows;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << matrix[i * cols + j] << " ";
        }
        cout << endl;
    }
}

#include <stdio.h>

float sumVector(float arr[], int size) {
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}

float meanVector(const vector<float> &matrix) {
    float sum = 0.0;
    for (int i = 0; i < matrix.size(); i++) {
        sum += matrix[i];
    }
    return sum / matrix.size();
}

void save_matrices(string file_name,
    const vector<float>& m1,
    unsigned int m1_rows,
    unsigned int m1_cols)
{
ofstream file(file_name);
if (file.is_open()) {
file << m1_rows << " " << m1_cols << endl;
for (int i = 0; i < m1_rows; i++) {
for (int j = 0; j < m1_cols; j++) {
file << m1[i * m1_cols + j] << " ";
}
file << endl;
}
}
else {
cout << "Unable to open file" << endl;
}
}
