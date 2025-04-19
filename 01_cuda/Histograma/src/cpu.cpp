#include "cpu.h"
#include "timer.h"
#include <vector>
#include <iostream>
#include <algorithm>

std::vector<int> cpu_histogram(const std::vector<float>& v, int num_bins, util::timer_pool& timers) {
    auto& timer = timers.cpu_add("CPU Histogram Computation", v.size());
    timer.do_start();

    std::vector<int> histogram(num_bins, 0);

    float min_val = *std::min_element(v.begin(), v.end());
    float max_val = *std::max_element(v.begin(), v.end());
    float range = max_val - min_val;

    std::cout << "CPU Min: " << min_val << ", Max: " << max_val << std::endl;

    for (float value : v) {
        int bin = std::min(num_bins - 1, std::max(0, static_cast<int>(((value - min_val) / range) * num_bins)));
        histogram[bin]++;
    }

    timer.do_stop();
    return histogram;
}


