#include <iostream>
#include <unistd.h>
#include "timer.h"

#include "kernel.h"

using namespace std;

int main(int argc, const char* argv[]) {

    auto& timer = util::timers.cpu_add("Total time");
    // call kernel
    kernel_wrapper();

    // sleep some seconds, otherwise 
    // device printf won't appear in the terminal
    cout << "sleeping for a few seconds..." << endl;
    sleep(1);

    timer.stop();
    util::timers.flush();
    return 0;
}
