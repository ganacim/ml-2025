#include <iostream>
#include <unistd.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"
#include "cpu.h"
#include <vector>

using namespace std;

int main(int argc, const char* argv[]) {
    //inicializando kernel e valores (tamanhos) a serem usados
    //kernel de media foi usado para blur
    int rows = 3;
    int cols = 2;
    int ksize = 2;
    vector<float> kernel = create_kernel(ksize);
    vector<float> m = create_random_matrix(rows, cols);

    //calculando convolucao2d com a cpu, cpu c/ openmp e gpu
    vector<float> result_naive_cpu = cpu_naive_conv2d(m, kernel, ksize, cols, rows);
    vector<float> result_openmp_cpu = openmp_conv2d(m, kernel, ksize, cols, rows);
    vector<float> result_m = cuda_convolution_template(m, kernel, rows, cols, ksize);
    

    //opcao de printar matrizes para checar resultados
    //print_matrix(result_naive_cpu, rows-ksize+1);
    //print_matrix(result_openmp_cpu, rows-ksize+1);
    //print_matrix(result_m, rows-ksize+1);
    //print_matrix(m, rows);

    //opcao de salvar matrizes que geram imagens
    //save_matrices("/impa/home/l/joao.crema/Desktop/ml-2025/01_cuda/original.txt", m, rows, cols);
    //save_matrices("/impa/home/l/joao.crema/Desktop/ml-2025/01_cuda/filtered.txt", result_m, rows-ksize+1, cols-ksize+1);
    util::timers.flush();
    sleep(0);
    return 0;
}