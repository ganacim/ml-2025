#include <stdio.h>
#include <cuda.h>

// Define block size (threads per block)
#define BLOCK_SIZE 16

// Kernel 1: Compute prefix sum within each block (Partial Scan)
__global__ void block_prefix_sum(int* d_in, int* d_out, int* d_block_sums, int n) {
    __shared__ int temp[BLOCK_SIZE];  // Shared memory for intra-block scan

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x; // Global index

    // Load input to shared memory if within bounds
    temp[tid] = (gid < n) ? d_in[gid] : 0;
    __syncthreads(); // não pesquei totalmente esse __syncthreads


    //soma
    for (int offset = 1; offset < blockDim.x; offset <<= 1) { 
        int t = (tid >= offset) ? temp[tid - offset] : 0;
        __syncthreads();
        temp[tid] += t;
        __syncthreads();
    }

    // Write partial sums to global memory
    if (gid < n) {
        d_out[gid] = temp[tid];
    }

    // Save the last element of each block for later adjustment
    if (tid == blockDim.x - 1) {
        d_block_sums[blockIdx.x] = temp[tid];
    }
}

// Kernel 2: Add the scanned block sums to each block
__global__ void add_block_sums(int* d_out, int* d_block_sums, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x > 0 && gid < n) {
        d_out[gid] += d_block_sums[blockIdx.x - 1];
    }
}

// Host function
void prefix_sum(int* h_in, int* h_out, int n) {
    int *d_in, *d_out, *d_block_sums, *d_block_sums_scan;

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Joga pra gpu
    cudaMalloc((void**)&d_in, n * sizeof(int));
    cudaMalloc((void**)&d_out, n * sizeof(int));
    cudaMalloc((void**)&d_block_sums, num_blocks * sizeof(int));
    cudaMalloc((void**)&d_block_sums_scan, num_blocks * sizeof(int));
    cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice);

    // Somas parciais
    block_prefix_sum<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, d_block_sums, n);

    // Volta pra cpu e soma lá
    int* h_block_sums = (int*)malloc(num_blocks * sizeof(int));
    cudaMemcpy(h_block_sums, d_block_sums, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
    h_block_sums[0] = h_block_sums[0];
    for (int i = 1; i < num_blocks; i++) {
        h_block_sums[i] += h_block_sums[i - 1];
    }

    // volta o h_block_sums pra gpu e termina o vetor
    cudaMemcpy(d_block_sums_scan, h_block_sums, num_blocks * sizeof(int), cudaMemcpyHostToDevice);
    add_block_sums<<<num_blocks, BLOCK_SIZE>>>(d_out, d_block_sums_scan, n);

    // Joga para fora da gpu
    cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Liberar toda a memória
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_block_sums);
    cudaFree(d_block_sums_scan);
}

int main() {
    const int N = 64; // Tamanho do vetor
    int h_in[N], h_out[N];

    // Definimos aqui o vetor de entrada 
    //Testei com tudo 1, tudo 2, tudo 3, tudo 0 e o vetor [0, 2, 3, 4, ..., 63]
    for (int i = 0; i < N; i++) {
        h_in[i] = i; 
    } 

    for (int i = 0; i < N; i++) {
        printf("%d ", h_in[i]);
    }; //print do vetor de entrada
    printf("\n");

    // Soma (inclusive)
    prefix_sum(h_in, h_out, N);

    // Print dos resultados
    for (int i = 0; i < N; i++) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    return 0;
}
