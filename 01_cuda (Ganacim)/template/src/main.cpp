#include <iostream>
#include <unistd.h>

#include "kernel.h"

using namespace std;

int main(int argc, const char* argv[]) {
    cout << "Hello World!" << endl;

    // call kernel
    kernel_wrapper();

    // sleep some seconds, otherwise 
    // device printf won't appear in the terminal
    cout << "sleeping for a few seconds..." << endl;
    sleep(10);

    return 0;
}