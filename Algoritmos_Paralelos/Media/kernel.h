#ifndef _KERNEL_H_
#define _KERNEL_H_
#include <chrono>

//void kernel_wrapper();
// Funções para CPU
float kernel_cpu(int *array, int tot_pontos){
    int soma_cpu=0;
    for(int i=0;i<tot_pontos;i++){
        soma_cpu+=array[i];
    }
    return soma_cpu; 
    
}

//Funções para GPU
// Define a kernel function, which is the entry point
// for execution on the GPU
__global__ 
void kernel(int *ptr_d,int tot_pontos, int *soma_d) {
    //media(ptr_d,tot_pontos, soma_d);
    int index = threadIdx.x;
    int stride = blockDim.x;
    //ptr_d +=index;
    for(int i=index;i<tot_pontos;i+=stride){
        atomicAdd(soma_d,ptr_d[i]);
    }
        
}


// Define a wrapper function, which launches the kernel
void kernel_wrapper(int *ptr_d,int tot_pontos, int *soma_d) {
    // Launch kernel with <<<block, thread>>> syntax
    kernel<<<1,32>>>(ptr_d,tot_pontos, soma_d);
}

//Função para checagem do tempo
long long getmicros() {
    //auto now = std::chrono::high_resolution_clock::now();

    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}
#endif