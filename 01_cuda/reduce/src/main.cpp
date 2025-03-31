#include <iostream>
#include <unistd.h>

#include "kernel.h"
#include "timer.h"
#include "gen.h"

using namespace std;

int main(int argc, const char* argv[]) {

    int n = 32*32*32*32-13;
    vector<float> v = create_random_vector(n);

    // call kernel
    kernel_wrapper(v);

    sleep(1);


    cpu_only(v);

    util::timers.flush();

    return 0;
}
