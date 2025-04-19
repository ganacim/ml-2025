#include <vector>
#include "Vector.h"


#include <stdio.h>
#include <iostream>

#include <ctime>
#include <random>

typedef std::mt19937 RNG;  
using namespace std;

#define BLOCK_SIZE 32



vector<float> create_random_vector(unsigned int cols)
{
    
    uint32_t seed = (uint32_t) time(0);    
    RNG rng(seed);
    normal_distribution<float> normal(0.0, 1.0);
    
    vector<float> matrix(cols);
    for (int j = 0; j < cols; j++) {
        matrix[j] = normal(rng);
    }
    return matrix;
}
#include <iostream>

