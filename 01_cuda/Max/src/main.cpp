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
    cout << "Usage: " << name << " --size:<vector_size> [--cpu] [--cuda] [--compare]" << endl;
    cout << "    --size:<vector_size>    Set the size of the vector (default: 1024)" << endl;
    cout << "    --cpu                   Run CPU version" << endl;
    cout << "    --cuda                  Run CUDA version" << endl;
    cout << "    --compare               Compare CPU and CUDA results" << endl;
}

int main(int argc, const char** argv) {
    int vector_size = 1024;  
    bool run_cpu = false, run_cuda = false, compare = false;

    
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];

        if (arg.find("--size:") == 0) {
            vector_size = stoi(arg.substr(7));
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

     //Quando descomentamos essa linha, conseguimos ver o vetor gerado.
    //cout << "Vetor Gerado: " << endl;
    //for (int i = 0; i < vector_size; i++) {
        //cout << v[i] << " ";
        //if ((i + 1) % 10 == 0) cout << endl;
    //}
    //cout << endl;

    
    util::timer_pool timers;

    
    float max_cpu = 0;
    if (run_cpu) {
        auto& cpu_timer = timers.cpu_add("CPU Execution", vector_size);
        cpu_timer.do_start();
        max_cpu = cpu_max_value(v, timers);  
        cpu_timer.do_stop();
        cout << "Máximo CPU: " << max_cpu << " (Tempo: " << cpu_timer.do_get_elapsed() << " ms)" << endl;
    }

    
    float max_cuda = 0;
    if (run_cuda) {
        auto& cuda_timer = timers.gpu_add("CUDA Execution", vector_size);
        cuda_timer.do_start();
        max_cuda = cuda_max_value(v, timers);
        cuda_timer.do_stop();
        cout << "Máximo CUDA: " << max_cuda << " (Tempo: " 
        << cuda_timer.do_get_elapsed() << " ms)" << endl;   
    }

    
    if (compare) {
        cout << "Diferença entre CPU e CUDA: " << fabs(max_cpu - max_cuda) << endl;
    }

    return 0;
}
