#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"
#include "cpu.h"
#include <vector>


using namespace std;

void save_pgm(const char* filename, const std::vector<float>& data, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        perror("Cannot open file");
        return;
    }

    
    fprintf(file, "P5\n%d %d\n255\n", width, height);


    for (int i = 0; i < width * height; i++) {
        unsigned char pixel = (unsigned char)(255.0f * data[i]);  
        fwrite(&pixel, sizeof(unsigned char), 1, file);
    }

    fclose(file);
}


int main(int argc, const char* argv[]) {
    //inicializando kernel e valores (tamanhos) a serem usados
    //kernel de media foi usado para blur
    int rows = 32*2*2*2;
    int cols = 32*2*2*2;
    int ksize = 32;
    vector<float> kernel = create_kernel(ksize);
    vector<float> m = create_random_matrix(rows, cols);

    //calculando convolucao2d com a cpu, cpu c/ openmp e gpu
    vector<float> result_naive_cpu = cpu_naive_conv2d(m, kernel, ksize, cols, rows);
    vector<float> result_openmp_cpu = openmp_conv2d(m, kernel, ksize, cols, rows);
    vector<float> result_m = cuda_convolution_template(m, kernel, rows, cols, ksize);

    //salvando imagem, para ser exibida
    //pegar a imagem no endereco especificado
    save_pgm("noisy_image.pgm", m, cols, rows);
    save_pgm("filtered_image.pgm", result_m, cols-ksize+1, rows-ksize+1);


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