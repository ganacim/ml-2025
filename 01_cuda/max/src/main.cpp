#include <iostream>
#include <unistd.h>
#include <vector>

#include "cpu.h"
#include "kernel.h"
#include "vector.h"
#include "timer.h"

#include <cstring>
#include <regex>
#include <cmath>

using namespace std;

int main(int argc, const char* argv[]) {
    // call kernel
    unsigned int vsize = 100;
    vector<float> v;

    v = create_random_vector(vsize);


    float max_valuecpunaive;
    float max_valuecpuopnmp;

    max_valuecpunaive = cpu_naive_maxvalue(v);
    max_valuecpuopnmp = openmp_maxvalue(v);
    kernel_wrapper(v);

    cout << max_valuecpunaive << endl;
    cout << max_valuecpuopnmp << endl;

    // sleep some seconds, otherwise 
    // device printf won't appear in the terminal
    util::timers.flush();
    sleep(0);
    return 0;
}