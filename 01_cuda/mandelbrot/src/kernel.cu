#include "kernel.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

// REFERENCE:
// https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set#Optimized_escape_time_algorithms
__global__ void mandelbrot(unsigned char *image, int max_iter, int height,
                           int width, float xmin, float xmax, float ymin,
                           float ymax) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= width or j >= height) {
    return;
  }

  float x0 = xmin + (xmax - xmin) * (i / (float)width);
  float y0 = ymin + (ymax - ymin) * (j / (float)height);
  float x2 = 0.0;
  float y2 = 0.0;
  float w = 0.0;
  float x;
  float y;

  int iter = 0;
  while (x2 + y2 <= 4.0 and iter < max_iter) {
    x = x2 - y2 + x0;
    y = w - x2 - y2 + y0;
    x2 = x * x;
    y2 = y * y;
    w = (x + y) * (x + y);
    iter = iter + 1;
  }
  unsigned char color = (unsigned char)((255 * iter) / max_iter);
  int index = (i * width + j) * 3;
  image[index] = color;
  image[index + 1] = color;
  image[index + 2] = color;
  __syncthreads();
}

// Reference: ChatGPT
void save_image(const string &filename, unsigned char *image, int width,
                int height) {
  ofstream file(filename, ios::binary);
  file << "P6\n" << width << " " << height << "\n255\n";
  file.write(reinterpret_cast<char *>(image), width * height * 3);
  file.close();
}

// Define a wrapper function, which launches the kernel
void kernel_wrapper() {
  int height = 2048;
  int width = 2048;

  float xmin = -2.0, xmax = 2.0, ymin = -2.0, ymax = 2.0;

  int block_size = 32;
  int max_iter = 10000;

  dim3 grid((height + block_size - 1) / block_size,
            (width + block_size - 1) / block_size);
  dim3 block(block_size, block_size);

  unsigned char *image;
  cudaMalloc(&image, height * width * sizeof(unsigned char) * 3);

  // Launch kernel with <<<block, thread>>> syntax
  mandelbrot<<<grid, block>>>(image, max_iter, height, width, xmin, xmax, ymin,
                              ymax);

  // Save Image
  vector<unsigned char> host_image(height * width * 3);

  cudaMemcpy(host_image.data(), image,
             height * width * sizeof(unsigned char) * 3,
             cudaMemcpyDeviceToHost);
  cudaFree(image);

  save_image("mandelbrot.ppm", host_image.data(), width, height);
  cout << "Image saved to mandelbrot.ppm" << endl;
}
