#ifndef CUDA_H
#define CUDA_H

#include <vector>
#include "timer.h"  

float cuda_max_value(const std::vector<float>& v, util::timer_pool& timers);

#endif  
