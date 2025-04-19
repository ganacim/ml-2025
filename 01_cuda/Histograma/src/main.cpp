#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "Vector.h"
#include "timer.h"
#include "cpu.h"
#include "cuda.h"

using namespace std;

void help(const char *name) {
    cout << "Usage: " << name << " --size:<vector_size> [--cpu] [--cuda] [--compare] [--bins:<num_bins>]" << endl;
}

int main(int argc, const char** argv) {
    int vector_size = 1024;
    int num_bins = 10;  
    bool run_cpu = false, run_cuda = false, compare = false;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];

        if (arg.find("--size:") == 0) {
            vector_size = stoi(arg.substr(7));
        } else if (arg.find("--bins:") == 0) {
            num_bins = stoi(arg.substr(7));
        } else if (arg == "--cpu") {
            run_cpu = true;
        } else if (arg == "--cuda") {
            run_cuda = true;
        } else if (arg == "--compare") {
            compare = true;
            run_cpu = true;
            run_cuda = true;
        } else {
            help(argv[0]);
            return 1;
        }
    }

    
    vector<float> v = create_random_vector(vector_size);  
    vector<int> hist_cpu(num_bins, 0), hist_cuda(num_bins, 0);

    util::timer_pool timers;

    
    if (run_cpu) {
        auto& cpu_timer = timers.cpu_add("CPU Execution", vector_size);
        cpu_timer.do_start();
        hist_cpu = cpu_histogram(v, num_bins, timers);
        cpu_timer.do_stop();
        cout << "Histograma CPU calculado em: " << cpu_timer.do_get_elapsed() << " ms" << endl;
    }

    
    if (run_cuda) {
        auto& cuda_timer = timers.gpu_add("CUDA Execution", vector_size);
        cuda_timer.do_start();
        hist_cuda = cuda_histogram(v, num_bins, timers);
        cuda_timer.do_stop();
        cout << "Histograma CUDA calculado em: " << cuda_timer.do_get_elapsed() << " ms" << endl;
    }

    
    if (compare) {
        cout << "Comparação entre CPU e CUDA:" << endl;
        for (int i = 0; i < num_bins; i++) {
            cout << "Bin " << i << ": CPU=" << hist_cpu[i] << ", CUDA=" << hist_cuda[i] << endl;
        }
    }

    return 0;
}