#ifndef _CUDA_H_
#define _CUDA_H_

#include <vector>

std::vector<float> cuda_multiplication(const std::vector<float>& m1,
                                    const std::vector<float>& m2,
                                    unsigned int m1_rows,
                                    unsigned int m1_cols,
                                    unsigned int m2_cols,
                                    unsigned int block_size,
                                    bool use_shared_memory);

#endif
