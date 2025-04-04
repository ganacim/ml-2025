#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <vector>
#include <string>

std::vector<float> create_random_matrix(unsigned int rows, unsigned int cols);

void print_matrix(const std::vector<float> &matrix, unsigned int rows);

void save_matrix(std::string file_name,
                    const std::vector<float>& m,
                    unsigned int m_rows,
                    unsigned int m_cols);

void save_matrices(std::string file_name,
                    const std::vector<float>& m1,
                    const std::vector<float>& m2,
                    unsigned int m1_rows,
                    unsigned int m1_cols,
                    unsigned int m2_cols);

bool load_matrices(std::string file_name,
                    std::vector<float>& m1,
                    std::vector<float>& m2,
                    unsigned int& m1_rows,
                    unsigned int& m1_cols,
                    unsigned int& m2_cols);

#endif