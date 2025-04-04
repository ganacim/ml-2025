#ifndef _CPU_H_
#define _CPU_H_

#include <vector>

// Naive matrix multiplication
std::vector<float> cpu_naive_multiplication(const std::vector<float>& m1,
                                        const std::vector<float>& m2,
                                        unsigned int m1_rows,
                                        unsigned int m1_cols,
                                        unsigned int m2_cols);

std::vector<float> openmp_multiplication(const std::vector<float>& m1,
                                    const std::vector<float>& m2,
                                    unsigned int m1_rows,
                                    unsigned int m1_cols,
                                    unsigned int m2_cols);

#endif