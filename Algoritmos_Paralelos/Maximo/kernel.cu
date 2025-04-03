

#include <stdio.h>

// Define a device function, which 
// can be called from a kernel and executes on the GPU
__device__ int device_function(float *x, int tot_pontos){
    printf("Hello CUDA World!\n");
    int index = threadIdx.x;
    int stride = blockDim.x;
    for(int i = index; i < tot_pontos; i+=stride){
        if (x*x+y*y<1){

        }
    }
    return 1;
} 

// Define a kernel function, which is the entry point
// for execution on the GPU
__global__ 
void kernel(int pontos_in,int tot_pontos) {
    device_function(pontos_in,tot_pontos);
        
}

// Define a wrapper function, which launches the kernel
void kernel_wrapper(int pontos_in,int tot_pontos) {
    // Launch kernel with <<<block, thread>>> syntax
    kernel<<<8,256>>>(pontos_in,tot_pontos);
}
int main(void){
    int pontos_in = 0;
    int tot_pontos = 1<<20;

    kernel_wrapper(pontos_in,tot_pontos);
    cudaDeviceSynchronize();
    return 0;
}