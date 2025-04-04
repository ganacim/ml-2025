#include <iostream>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "kernel.h"
#include <chrono>

using namespace std;


#define IM_SIZE_COOR 1024

using namespace std::chrono;


float return_color_mandelbrot_cpu(int x0, int y0) {
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


int main(int argc, const char* argv[]) {
    cout << "Calculating the Mandelbrot set" << endl;


    // call kernel
    kernel_wrapper();


    // sleep some seconds, otherwise 
    // device printf won't appear in the terminal
    cout << "sleeping for a few seconds..." << endl;
    sleep(2);

    cout << "Calculating the Mandelbrot set in CPU" << endl;
    auto start2 = high_resolution_clock::now();

    int width = IM_SIZE_COOR, height = IM_SIZE_COOR;  // Grid size


    vector<vector<float>> data(height, vector<float>(width));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            data[y][x] = return_color_mandelbrot_cpu(x,y);  // Random float between 0 and 1
        }
    }

    auto end2 = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(end2 - start2);
    cout << "Execution Time in CPU: " << duration2.count() << " us" << endl;


    return 0;
}