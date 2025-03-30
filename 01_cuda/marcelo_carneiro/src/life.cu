#include <stdio.h>
#include <cuda.h>
#include <unistd.h>  




#define N 32
#define BLOCK_SIZE 32 // BLOCK_SIZE é para ser igual a N

__global__ void game_of_life_step(int* d_curr, int* d_next) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return; // trabalhar só dentro de uma matriz de tamanho N por N

    int alive_neighbors = 0;

    for (int dr = -1; dr <= 1; dr++) { //dr para mudança na row e dc para mudança na column
        for (int dc = -1; dc <= 1; dc++) {
            if (dr == 0 && dc == 0) continue; // Para não contar a si mesmo

            int r = row + dr;
            int c = col + dc;

            // na fronteira, usar os vizinhos de fora como mortos
            if (r >= 0 && r < N && c >= 0 && c < N) {
                alive_neighbors += d_curr[r * N + c];
            }
        }
    }

    int cell = d_curr[row * N + col];
    if (cell == 1) {
        if (alive_neighbors == 2 || alive_neighbors == 3)
            d_next[row * N + col] = 1;
        else
            d_next[row * N + col] = 0;
    } else {
        if (alive_neighbors == 3)
            d_next[row * N + col] = 1;
        else
            d_next[row * N + col] = 0;
    }
}


// Tabuleiro
void ani_board(int* board, int gen, int speed) {
    printf("\033[H\033[2J");  // Tentei outras coisas pra limpar a tela, mas ficou bugado; principalmente aumentando o tamanho da grid. Usei a sol. daqui: https://www.reddit.com/r/C_Programming/comments/1ez0vhl/whats_the_most_efficient_way_to_clear_the_terminal/
    printf("Generation %d:\n", gen);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%s ", board[i * N + j] ? "o" : " ");
        }
        printf("\n");
    }
    usleep(speed); //espera
}


int main() {
    int h_board[N * N] = {0}; // Host board
    int* d_curr;  // atual (device) (gpu)
    int* d_next;  // próximo (device) (gpu)

    // glider 1
    h_board[1 * N + 2] = 1;
    h_board[2 * N + 3] = 1;
    h_board[3 * N + 1] = 1;
    h_board[3 * N + 2] = 1;
    h_board[3 * N + 3] = 1;

    // glider 2
    h_board[N * N - 2] = 1;
    h_board[(N-1) * N - 3] = 1;
    h_board[(N-2) * N - 1] = 1;
    h_board[(N-2) * N - 2] = 1;
    h_board[(N-2) * N - 3] = 1;

    // loop simples
    h_board[4*N - 4] = 1;
    h_board[4*N - 5] = 1;
    h_board[4*N - 6] = 1;


    // Prepara tudo na gpu
    cudaMalloc(&d_curr, N * N * sizeof(int));
    cudaMalloc(&d_next, N * N * sizeof(int));
    cudaMemcpy(d_curr, h_board, N * N * sizeof(int), cudaMemcpyHostToDevice);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Calculo + animação
    for (int gen = 0; gen < 70; gen++) {
        cudaMemcpy(h_board, d_curr, N * N * sizeof(int), cudaMemcpyDeviceToHost);
        ani_board(h_board, gen, 400000);

        game_of_life_step<<<blocks, threads>>>(d_curr, d_next);

        int* tmp = d_curr;
        d_curr = d_next;
        d_next = tmp;
    }

    // Limpa a memória
    cudaFree(d_curr);
    cudaFree(d_next);

    return 0;
}
