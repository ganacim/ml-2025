
#include <stdio.h>
#include <iostream>
#include "kernel.h"


int main(void){
    
    //int pontos_in = 0;
    int tot_pontos = 256;

    int array[tot_pontos];
    int *ptr_d; 
    int s,*soma_d;
    s = 0;
    
    for(int i = 0; i<tot_pontos;i++){
        array[i] = i;
    }

    // Começo do processo da GPU
    using namespace std::chrono;
    long long t0,t1,t2;
    t0 = getmicros();
    getmicros();
    cudaMalloc((void**)&ptr_d,tot_pontos*sizeof(int));
    cudaMalloc((void**)&soma_d,sizeof(int));

    cudaMemcpy(ptr_d,array,tot_pontos*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(soma_d,&s,sizeof(int),cudaMemcpyHostToDevice);
    long long t01 = getmicros();
    kernel_wrapper(ptr_d,tot_pontos,soma_d);
    long long t02 = getmicros();
    cudaMemcpy(&s,soma_d,sizeof(int),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    float media_gpu = static_cast<float>(s)/tot_pontos;
    std::cout <<"Média da gpu é "<< media_gpu<< std::endl;

    cudaFree(ptr_d);
    cudaFree(soma_d);
    t1 = getmicros();

    float media_host;
    media_host = kernel_cpu(array,tot_pontos)/tot_pontos;
    std::cout <<"Média da cpu é "<< media_host<< std::endl;
    t2 = getmicros();
    //long long dif1,dif2;

    // long long d1,d2;
    // d1 = t1-t0;
    // d2 = t2-t1;
    auto dif1 = (t1 - t0);
    auto dif2 = (t2 - t1);
    auto dif3 = (t02 - t01);
    std::cout <<"Tempo em micros (GPU + MemAlloc e Copies) é "<< dif1 << std::endl;
    std::cout <<"Tempo em micros (GPU + chamada do kernel) é "<< dif3 << std::endl;
    std::cout <<"Tempo em micros (CPU) é "<< dif2 << std::endl;

    return 0;
}