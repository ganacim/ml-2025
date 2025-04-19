#include "cpu.h"
#include "timer.h"
#include <algorithm>  
#include <vector>


float cpu_max_value(const std::vector<float>& v, util::timer_pool& timers) {
    auto& timer = timers.cpu_add("CPU Max Value", v.size());  
    timer.do_start();

    float max_val = *std::max_element(v.begin(), v.end());

    timer.do_stop();
    return max_val;
}

