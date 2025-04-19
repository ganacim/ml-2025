#ifndef _CPU_H
#define _CPU_H

#include <vector>
#include "timer.h"  

float cpu_max_value(const std::vector<float>& v, util::timer_pool& timers);

#endif
