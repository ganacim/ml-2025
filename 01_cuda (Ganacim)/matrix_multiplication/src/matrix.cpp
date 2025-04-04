#include "matrix.h"

#include <iostream>
#include <fstream>
#include <ctime>
#include <random>

typedef std::mt19937 RNG;  // Mersenne Twister with a popular choice of parameters

using namespace std;

vector<float> create_random_matrix(unsigned int rows, unsigned int cols)
{
    // Create a normal distribution with mean 0 and standard deviation 1
    uint32_t seed = (uint32_t) time(0);    
    RNG rng(seed);
    normal_distribution<float> normal(0.0, 1.0);
    // Create a matrix of size matrix_size x matrix_size with random values
    vector<float> matrix(rows * cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = normal(rng);
        }
    }
    return matrix;
}

void print_matrix(const vector<float> &matrix, unsigned int rows)
{
    unsigned int cols = matrix.size() / rows;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << matrix[i * cols + j] << " ";
        }
        cout << endl;
    }
}

void save_matrix(string file_name,
                    const vector<float>& m,
                    unsigned int m_rows,
                    unsigned int m_cols)
{
    ofstream file(file_name);
    if (file.is_open()) {
        file << m_rows << " " << m_cols << endl;
        for (int i = 0; i < m_rows; i++) {
            for (int j = 0; j < m_cols; j++) {
                file << m[i * m_cols + j] << " ";
            }
            file << endl;
        }
        file.close();
    }
    else {
        cout << "Unable to open file" << endl;
    }
}

void save_matrices(string file_name,
                    const vector<float>& m1,
                    const vector<float>& m2,
                    unsigned int m1_rows,
                    unsigned int m1_cols,
                    unsigned int m2_cols)
{
    unsigned int m2_rows = m1_cols;
    ofstream file(file_name);
    if (file.is_open()) {
        file << m1_rows << " " << m1_cols << endl;
        for (int i = 0; i < m1_rows; i++) {
            for (int j = 0; j < m1_cols; j++) {
                file << m1[i * m1_cols + j] << " ";
            }
            file << endl;
        }
        file << m2_rows << " " << m2_cols << endl;
        for (int i = 0; i < m2_rows; i++) {
            for (int j = 0; j < m2_cols; j++) {
                file << m2[i * m2_cols + j] << " ";
            }
            file << endl;
        }
        file.close();
    }
    else {
        cout << "Unable to open file" << endl;
    }
}

bool load_matrices(string file_name,
                    vector<float>& m1,
                    vector<float>& m2,
                    unsigned int& m1_rows,
                    unsigned int& m1_cols,
                    unsigned int& m2_cols)
{
    ifstream file(file_name);
    if (file.is_open()) {
        file >> m1_rows >> m1_cols;
        m1.resize(m1_rows * m1_cols);
        for (int i = 0; i < m1_rows; i++) {
            for (int j = 0; j < m1_cols; j++) {
                file >> m1[i * m1_cols + j];
            }
        }
        unsigned int m2_rows;
        file >> m2_rows >> m2_cols;
        m2.resize(m2_rows * m2_cols);
        for (int i = 0; i < m2_rows; i++) {
            for (int j = 0; j < m2_cols; j++) {
                file >> m2[i * m2_cols + j];
            }
        }
        if (m1_cols != m2_rows) {
            cout << "Invalid matrix sizes: [" << m1_rows << ", " << m1_cols 
                    << "] and [" << m2_rows << ", " << m2_cols << "]" << endl;
            return false;
        }
        file.close();
        return true;
    }
    else {
        cout << "Unable to open file" << endl;
        return false;
    }
}
