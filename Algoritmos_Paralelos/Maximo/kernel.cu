
#include <stdio.h>
#include <iostream>

// Define a device function, which 
// can be called from a kernel and executes on the GPU
__device__ int media(int *ptr, int tot_pontos, int *soma){
    //printf("Hello CUDA World!\n");
    int index = threadIdx.x;
    //int stride = blockDim.x;
    ptr +=index;
    *soma += *ptr/tot_pontos;
    
    // for(int i = index; i < tot_pontos; i+=stride){
    //     if (x*x+y*y<1){

    //     }
    // }
    return 1;
} 

// Define a kernel function, which is the entry point
// for execution on the GPU
__global__ 
void kernel(int *ptr,int tot_pontos, int *soma) {
    media(ptr,tot_pontos, soma);
        
}

// Define a wrapper function, which launches the kernel
void kernel_wrapper(int *ptr,int tot_pontos, int *soma) {
    // Launch kernel with <<<block, thread>>> syntax
    kernel<<<1,32>>>(ptr,tot_pontos, soma);
}
int main(void){
    //int pontos_in = 0;
    int tot_pontos = 32;

    int array[tot_pontos];
    int *ptr = array; 
    int s,*soma;
    soma = &s;
    for(int i = 0; i<tot_pontos;i++){
        array[i] = 8;
    }
    kernel_wrapper(ptr,tot_pontos,soma);
    std::cout << s;
    cudaDeviceSynchronize();
    
    return 0;
}