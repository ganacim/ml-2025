#include <iostream>
#include <unistd.h>
#include <vector>
#include <complex>

//#include "kernel.h"
#include <chrono>
#include <GL/glut.h>

using namespace std;
const int WIDTH = 400;
const int HEIGHT = 400;
const int MAX_ITER = 20;
std::vector<unsigned char> pixelData(WIDTH * HEIGHT * 3);  // RGB format
GLuint textureID;

float scale = 4.0f;
float panX = -0.4f, panY = 0.0f;

int calc_mandelbrot(float x,float y){
    x = ( x/WIDTH - 0.5) * scale + panX;
    y = ( y/HEIGHT - 0.5) * scale + panY;

    complex<float> z(0.0, 0.0), c(x,y);
    int i = 0; 

    while(abs(z) < 2 && i < MAX_ITER){
        z = z*z + c;
        i++;
    }

    if (i < 4){
        i = 3;
    } else if(i == MAX_ITER){
        i = 10;
    } else{
        i = i % 7;
    }

    //cout << x << " " << y << ":" << i << endl;
    return i;   
}

int *colors = new int[WIDTH*HEIGHT];
void generateImage() {
    //kernel_wrapper(colors, WIDTH, HEIGHT, scale, panX, panY);
    for (float y = 0; y < HEIGHT; ++y) {
        for (float x = 0; x < WIDTH; ++x) {

            int index = int(y * WIDTH + x)*3;  // pixel position // 3 channels per pixel (RGB)
            int color_key = calc_mandelbrot(x,y);

            if (color_key == 3 ) {  // Blue
                pixelData[index] = 0; pixelData[index + 1] = 0; pixelData[index + 2] = 255;
            } else if (color_key == 4) {  // Cyan
                pixelData[index] = 0  ; pixelData[index + 1] = 255; pixelData[index + 2] = 255;
            } else if (color_key == 5) {  // Green
                pixelData[index] = 0  ; pixelData[index + 1] = 255; pixelData[index + 2] = 0;
            } else if (color_key == 6) {  // Yellow
                pixelData[index] = 255; pixelData[index + 1] = 255; pixelData[index + 2] = 0;
            } else if (color_key == 0) {  // Orange
                pixelData[index] = 255; pixelData[index + 1] = 128; pixelData[index + 2] = 0;
            } else if (color_key == 1) {  // Red
                pixelData[index] = 255; pixelData[index + 1] = 0; pixelData[index + 2] = 0;
            } else if (color_key == 2) {  // Magenta
                pixelData[index] = 255; pixelData[index + 1] = 0; pixelData[index + 2] = 255;
            } else {  // Black
                pixelData[index] = 0; pixelData[index + 1] = 0; pixelData[index + 2] = 0;

            } 
        }
    }
}

// Load the image as an OpenGL texture
void loadTexture() {
    glGenTextures(1, &textureID);  // Generate texture ID
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Upload pixel data to the texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, pixelData.data());

    glBindTexture(GL_TEXTURE_2D, 0);  // Unbind texture
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_TEXTURE_2D);

    auto start_cpu = chrono::high_resolution_clock::now();
    generateImage();
    loadTexture();
    auto end_cpu = chrono::high_resolution_clock::now();
    auto delta_cpu = chrono::duration<double>(end_cpu - start_cpu).count();
    cout << "Frame gerado em " << delta_cpu << " segundos. FPS:" << 1/delta_cpu << endl;

    glBindTexture(GL_TEXTURE_2D, textureID);

    glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(-1, -1);
        glTexCoord2f(1, 0); glVertex2f( 1, -1);
        glTexCoord2f(1, 1); glVertex2f( 1,  1);
        glTexCoord2f(0, 1); glVertex2f(-1,  1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glFlush();
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 'e': scale *= 0.9f; break;  // Zoom in
        case 'q': scale *= 1.1f; break;  // Zoom out
        case 'w': panY += scale*0.1f; break;  // Move up
        case 's': panY -= scale*0.1f; break;  // Move down
        case 'a': panX -= scale*0.1f; break;  // Move left
        case 'd': panX += scale*0.1f; break;  // Move right
        case 'r': panX = -0.4; panY = 0; scale = 4.0; break; // resets everything
        case 27: exit(0); break;        // Escape key exits
    }
    glutPostRedisplay();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Mandelbrot Set");

    glClearColor(1.0, 1.0, 1.0, 1.0);  // White background
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-1, 1, -1, 1);

    // Done in display func
    //generateImage();  // Fill pixel data
    //loadTexture();    // Convert to texture

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMainLoop();

    return 0;
}