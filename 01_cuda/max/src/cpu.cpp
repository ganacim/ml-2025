#include "cpu.h"
#include "timer.h"
#include <iostream>
#include <vector>
#include <limits>
using namespace std;

#include <nvtx3/nvtx3.hpp>

float cpu_naive_maxvalue(const std::vector<float>& v){
    float max_value = -std::numeric_limits<float>::infinity();
    auto& timer = util::timers.cpu_add("CPU Naive Max Value");
    nvtx3::mark("Begin CPU Max Value finder");

    for (float num : v){
        if (num > max_value){
            max_value = num;
        }
    }
    timer.stop();
    return max_value;
}

float openmp_maxvalue(const std::vector<float>& v)
{
    auto& timer = util::timers.cpu_add("CPU (OpenMP) Max Value");
    float max_value = -std::numeric_limits<float>::infinity();

    #pragma omp parallel for reduction(max: max_value)
    for (int i = 0; i < v.size(); i++) {
        if (v[i] > max_value){
            max_value = v[i];
        }
    }
    timer.stop();
    return max_value;
}

