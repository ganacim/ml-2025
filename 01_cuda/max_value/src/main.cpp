#include <iostream>
#include <unistd.h>
#include <vector>
#include <random>
#include <cfloat> 
#include <chrono>

#include <fstream>

#include "kernel.h"

using namespace std::chrono;


using namespace std;
typedef std::mt19937 RNG;  // Mersenne Twister with a popular choice of parameters

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


int main(int argc, const char* argv[]) {


    vector<float> v = create_random_vector(10000000);
    cout << "v.size(): " << v.size() << endl;
    cout << "Calculating the maximum in GPU" << endl;
    // call kernel
    
    kernel_wrapper(v);
    
    // sleep some seconds, otherwise 
    // device printf won't appear in the terminal
    cout << "sleeping for a few seconds..." << endl;
    sleep(2);

    cout << "Calculating the maximum in CPU" << endl;
    auto start2 = high_resolution_clock::now();
    float max_val = -FLT_MAX;

    for (float x: v){
        if (x > max_val){
            max_val = x;
        }
    }
    auto end2 = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(end2 - start2);
    cout << "Execution Time in CPU: " << duration2.count() << " us" << endl;
    cout << "The maximum value in CPU was: " << max_val << endl;




    return 0;
}