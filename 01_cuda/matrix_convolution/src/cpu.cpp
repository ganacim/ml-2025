#include "cpu.h"
#include "timer.h"
#include<iostream>
#include <nvtx3/nvtx3.hpp>

using namespace std;

// Naive conv2d
vector<float> cpu_naive_conv2d(const std::vector<float>& m,
                            const std::vector<float>& kern,
                            unsigned int ksize,
                            unsigned int m_cols,
                            unsigned int m_rows){
    auto& timer = util::timers.cpu_add("CPU Conv2d");

    int rcols = m_cols - ksize + 1;
    int rrows = m_rows - ksize + 1;
    vector<float> result(rcols * rrows);
    //nvtx3::mark("Begin CPU Conv2d");
    for (int i = 0; i < rcols; i++) {
        //nvtx3::scoped_range r("Row " + to_string(i));
        for (int j = 0; j < rrows; j++) {
            float out_pixel = 0.0f;
            for (int x = 0; x < ksize; x++) {
                for (int y = 0; y<ksize; y++){
                    out_pixel += m[(j+y) * m_cols + i+x] * kern[y * ksize + x];
                }
            }
            result[j * rcols + i] = out_pixel;
        }
    }
    timer.stop();
    return result;
}

// OpenMP Conv2d
vector<float> openmp_conv2d(const std::vector<float>& m,
                            const std::vector<float>& kern,
                            unsigned int ksize,
                            unsigned int m_cols,
                            unsigned int m_rows){
    auto& timer = util::timers.cpu_add("CPU (OpenMP) Conv2d");
    int rcols = m_cols - ksize + 1;
    int rrows = m_rows - ksize + 1;
    vector<float> result(rcols * rrows);
    nvtx3::mark("Begin CPU Conv2d");
    #pragma omp parallel for
    for (int i = 0; i < rcols; i++) {
        nvtx3::scoped_range r("Row " + to_string(i));
        for (int j = 0; j < rrows; j++) {
            float out_pixel = 0.0f;
            for (int x = 0; x < ksize; x++) {
                for (int y = 0; y<ksize; y++){
                    out_pixel += m[(j+y) * m_cols + i+x] * kern[y * ksize + x];
                }
            }
            result[j * rcols + i] = out_pixel;
        }
    }
    timer.stop();
    return result;
}

