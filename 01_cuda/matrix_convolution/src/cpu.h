#ifndef _CPU_H_
#define _CPU_H_

#include <vector>

// Naive matrix multiplication
std::vector<float> cpu_naive_conv2d(const std::vector<float>& m,
                            const std::vector<float>& kern,
                            unsigned int ksize,
                            unsigned int m_cols,
                            unsigned int m_rows);

std::vector<float> openmp_conv2d(const std::vector<float>& m,
                            const std::vector<float>& kern,
                            unsigned int ksize,
                            unsigned int m_cols,
                            unsigned int m_rows);


#endif