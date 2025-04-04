#include "cpu.h"
#include "timer.h"

#include <nvtx3/nvtx3.hpp>

using namespace std;

// Naive matrix multiplication
vector<float> cpu_naive_multiplication(const std::vector<float>& m1,
                                    const std::vector<float>& m2,
                                    unsigned int m1_rows,
                                    unsigned int m1_cols,
                                    unsigned int m2_cols)
{
    auto& timer = util::timers.cpu_add("CPU Multiplication");

    vector<float> result(m1_rows * m2_cols);
    nvtx3::mark("Begin CPU Multiplication");
    for (int i = 0; i < m1_rows; i++) {
        nvtx3::scoped_range r("Row " + to_string(i));
        for (int j = 0; j < m2_cols; j++) {
            float sum = 0.0;
            for (int k = 0; k < m1_cols; k++) {
                sum += m1[i * m1_cols + k] * m2[k * m2_cols + j];
            }
            result[i * m2_cols + j] = sum;
        }
    }
    timer.stop();
    return result;
}

vector<float> openmp_multiplication(const std::vector<float>& m1,
                                    const std::vector<float>& m2,
                                    unsigned int m1_rows,
                                    unsigned int m1_cols,
                                    unsigned int m2_cols)
{
    auto& timer = util::timers.cpu_add("CPU (OpenMP) Multiplication");

    vector<float> result(m1_rows * m2_cols);
    #pragma omp parallel for
    for (int i = 0; i < m1_rows; i++) {
        nvtx3::scoped_range r("Row " + to_string(i));
        for (int j = 0; j < m2_cols; j++) {
            float sum = 0.0;
            for (int k = 0; k < m1_cols; k++) {
                sum += m1[i * m1_cols + k] * m2[k * m2_cols + j];
            }
            result[i * m2_cols + j] = sum;
        }
    }
    timer.stop();
    return result;
}

