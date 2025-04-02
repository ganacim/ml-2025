#include <iostream>
#include <unistd.h>

#include "kernel.h"
#include "timer.h"
#include "gen.h"

using namespace std;

int main(int argc, const char* argv[]) {

    int n = 32*32*32*32*32*32;
    vector<float> v = create_random_vector(n);

    kernel_wrapper(v, -4.0, 4.0, 32);

    sleep(1);


    //cpu_only(v);

    util::timers.flush();

    return 0;
}
