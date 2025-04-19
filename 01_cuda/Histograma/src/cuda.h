#ifndef CUDA_H
#define CUDA_H

#include <vector>
#include "timer.h"

std::vector<int> cuda_histogram(const std::vector<float>& v, int num_bins, util::timer_pool& timers);

#endif 