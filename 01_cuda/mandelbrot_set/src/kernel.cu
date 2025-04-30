#include <vector>
#include <cfloat> 
#include "kernel.h"
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#include <ctime>
#include <random>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef std::mt19937 RNG;  // Mersenne Twister with a popular choice of parameters

using namespace std::chrono;

using namespace std;

#define BLOCK_SIZE 32
#define IM_SIZE_COOR 2048*2

// Function to save a 2D array as a grayscale PNG
void save_image(const vector<vector<float>> &data, const string &filename) {
    int width = data[0].size();
    int height = data.size();

    // Convert floating-point values (0-1) to grayscale (0-255)
    vector<unsigned char> image(width * height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            image[y * width + x] = static_cast<unsigned char>(data[y][x] * 255);
        }
    }

    // Save the image using stb_image_write
    stbi_write_png(filename.c_str(), width, height, 1, image.data(), width);
    cout << "Image saved as " << filename << endl;
}

// Define a function for each grid point computation
__device__ float return_color_mandelbrot(int x0, int y0) {
    int iteration = 0;
    int max_iteration = 10000;
    float coordinatex0 = (float(x0)*(2.47))/(IM_SIZE_COOR) - 2;
    float coordinatey0 = (float(y0)*(2.24))/(IM_SIZE_COOR) - 1.12;
    float x = 0;
    float y =0;
    while (x*x+y*y <= 4 && iteration < max_iteration){
        float x_temp = x*x-y*y+coordinatex0;
        y = 2*x*y+coordinatey0;
        x = x_temp;
        iteration++;
    }
    float pixel_intensity = float(iteration)/max_iteration;

    return pixel_intensity;
}

__global__ void compute_grid(float *d_grid, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure we are within bounds
    if (x < width && y < height) {
        int index = y * width + x;  // Convert 2D index to 1D for memory access
        d_grid[index] = return_color_mandelbrot(x,y);
    }
}

void kernel_wrapper() {

    int width = IM_SIZE_COOR, height = IM_SIZE_COOR;  // Grid size
    float *h_grid = new float[width * height];

    float *d_grid;
    size_t size = width * height * sizeof(float);

    // Allocate memory on GPU
    cudaMalloc(&d_grid, size);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width+BLOCK_SIZE-1) / BLOCK_SIZE, 
                 (height+BLOCK_SIZE-1) / BLOCK_SIZE);

    

    auto start = high_resolution_clock::now();
    // Launch the kernel
    compute_grid<<<gridDim, blockDim>>>(d_grid, width, height);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    cout << "Execution Time in GPU: " << duration.count() << " us" << endl;

    // Copy results back to CPU
    cudaMemcpy(h_grid, d_grid, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_grid);

    vector<vector<float>> data(height, vector<float>(width));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x; 
            data[y][x] = h_grid[index];  // Random float between 0 and 1
        }
    }

    save_image(data, "mandelbrot.png");

    cout << "Saved image Mandelbrot_set" << endl;



}   


