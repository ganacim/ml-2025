#ifndef CPU_H
#define CPU_H

#include <vector>
#include "timer.h"

std::vector<int> cpu_histogram(const std::vector<float>& v, int num_bins, util::timer_pool& timers);

#endif 