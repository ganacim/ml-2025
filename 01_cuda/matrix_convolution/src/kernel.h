#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <vector>

std::vector<float> cuda_convolution_template(const std::vector<float>& m,
    const std::vector<float>& k,
    unsigned int rows,
    unsigned int cols,
    unsigned int kernel_size);

#endif
