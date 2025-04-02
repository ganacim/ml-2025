#include "vector.h"
#include <random>
#include <vector>
typedef std::mt19937 RNG;  // Mersenne Twister with a popular choice of parameters

using namespace std;

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