#include <iostream>
#include <unistd.h>

#include "kernel.h"

using namespace std;

int main(int argc, const char* argv[]) {

    // call kernel
    kernel_wrapper();

    // sleep some seconds, otherwise 
    // device printf won't appear in the terminal

    return 0;
}
