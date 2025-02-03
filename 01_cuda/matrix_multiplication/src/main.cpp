#include <iostream>
#include <unistd.h>
#include <cstring>
#include <regex>
#include <cmath>

#include <nvtx3/nvtx3.hpp>

#include "cuda.h"
#include "matrix.h"
#include "cpu.h"
#include "timer.h"

using namespace std;

void help(const char *name){
    cout << "Usage: " << name << " " << endl;
    cout << "    --size:<m1_rows>:<m1_cols>:<m1_cols> or --size:<matrix_size>" << endl;
    cout << "           example: --size:1024:1024:1024 or --size:1024" << endl;
    cout << "    --print -- prints all matrices." << endl;
    cout << "    --save:<output_file> -- saves matrices M1 and M2 to file." << endl;
    cout << "    --load:<input_file> -- load matrices M1 and M2 from file." << endl;
    cout << "    --cpu[:<output_file>] -- run CPU version and save result to file (optional)." << endl;
    cout << "    --openmp[:<output_file>] -- run CPU (with OpenMP support) version and save result to file (optional)." << endl;
    cout << "    --cuda[:<output_file>] -- run CUDA version and save result to file (optional)." << endl;
    cout << "    --cudablock[:<output_file>] -- run CUDA block version and save result to file (optional)." << endl;
    cout << "    --compare -- compare results from CPU, OpenMP, and CUDA." << endl;
}

float max_norm(const vector<float>& m1, const vector<float>& m2){
    float max_diff = 0.0;
    for (int i = 0; i < m1.size(); i++){
        max_diff = max(max_diff, abs(m1[i] - m2[i]));
    }
    return max_diff;
}

int match(const string& arg, const char *pattern, smatch &m){
    const regex re_str{string(pattern)};
    regex_match(arg, m, re_str);
    return m.size();
}

int main(int argc, const char* argv[]) {
    auto& timer = util::timers.cpu_add("Total time");
    cout << "# Matrix Multiplication." << endl;
    // Variables for holding parameters
    // size of the matrices
    unsigned int m1_rows = 1024, m1_cols = 1024, m2_cols = 1024;
    // save and load
    bool save_matrix_flag = false;
    bool load_matrix_flag = false;
    string output_file_name = "output.txt";
    string input_file_name = "input.txt";
    // running options
    bool run_cpu_flag = false;
    bool save_cpu_result_flag = false;
    string cpu_result_file = "cpu_result.txt";
    bool run_openmp_flag = false;
    bool save_openmp_result_flag = false;
    string openmp_result_file = "openmp_result.txt";
    bool run_cuda_flag = false;
    bool save_cuda_result_flag = false;
    string cuda_result_file = "cuda_result.txt";
    bool run_cudablock_flag = false;
    bool save_cudablock_result_flag = false;
    unsigned int cuda_block_size = 32;
    string cudablock_result_file = "cudablock_result.txt";
    // other options
    bool print_matrix_flag = false;
    bool run_compare_flag = false;
    // Matrices
    vector<float> m1, m2, cpu_result, openmp_result, cuda_result, cudablock_result;
    // Parse arguments
    if (argc <= 1){
        help(argv[0]);
        exit(1);
    }
    for (int i = 0; i < argc; i++){
        if (strcmp(argv[i], "-h") == 0 ||
                strcmp(argv[i], "--help") == 0){
            help(argv[0]);
            exit(0);
        }
    }
    for (int i = 1; i < argc; i++){
        const char *value;
        std::smatch m;
        const std::string arg(argv[i]);
        // --size:m1_rows // m1_rows = m1_cols = m2_cols
        if (match(arg, R"~(--size:(\d+))~", m) == 2){
            // https://stackoverflow.com/questions/56710024/what-is-a-raw-string
            m1_rows = m1_cols = m2_cols = std::stoi(m[1]);
        }
        // --size:m1_rows:m1_cols:m2_cols
        else if (match(arg, R"~(--size:(\d+):(\d+):(\d+))~", m) == 4){
            m1_rows = std::stoi(m[1]);
            m1_cols = std::stoi(m[2]);
            m2_cols = std::stoi(m[3]);
        }
        // --print
        else if (match(argv[i], "--print", m)){
            print_matrix_flag = true;
        }
        // --save:output_matrix_file
        else if (match(argv[i], R"~(--save:([\w\-_\/]+(?:\.\w+)?))~", m)){
            save_matrix_flag = true;
            output_file_name = m[1].str();
        }
        // --load:input_matrix_file
        else if (match(argv[i], R"~(--load:([\w\-_\/]+(?:\.\w+)?))~", m)){
            load_matrix_flag = true;
            input_file_name = m[1].str();
        }
        // --cpu[:output_file]
        else if (match(arg, R"~(--cpu(?::([\w\-_\/]+(?:\.\w+)?))?)~", m)){
            run_cpu_flag = true;
            if (m.size() == 2 and m[1].str().size() > 0){
                save_cpu_result_flag = true;
                cpu_result_file = m[1].str();
            }
        }
        // --openmp[:output_file]
        else if (match(arg, R"~(--openmp(?::([\w\-_\/]+(?:\.\w+)?))?)~", m)){
            run_openmp_flag = true;
            if (m.size() == 2 and m[1].str().size() > 0){
                save_openmp_result_flag = true;
                openmp_result_file = m[1].str();
            }
        }
        // --cuda[:output_file]
        else if (match(arg, R"~(--cuda(?::([\w\-_\/]+(?:\.\w+)?))?)~", m)){
            run_cuda_flag = true;
            if (m.size() == 2 and m[1].str().size() > 0){
                save_cuda_result_flag = true;
                cuda_result_file = m[1].str();
            }
        }
        // --cudablock[:output_file]
        else if (match(arg, R"~(--cudablock(?::([\w\-_\/]+(?:\.\w+)?))?)~", m)){
            run_cudablock_flag = true;
            if (m.size() == 2 and m[1].str().size() > 0){
                save_cudablock_result_flag = true;
                cudablock_result_file = m[1].str();
            }
        }
        // --blocksize:<block_size>
        else if (match(arg, R"~(--blocksize:(\d+))~", m) == 2){
            cuda_block_size = std::stoi(m[1]);
        }
        // --compare
        else if (match(arg, "--compare", m)){
            run_compare_flag = true;
        }
        // Invalid argument
        else {
            cout << "! Invalid argument: " << argv[i] << endl;
            exit(1);
        }
    }
    // Sleep so the profiler can attach
    cout << "Sleeping for 1 second." << endl;
    sleep(1);
    //
    { // Scope for nvtx3
        nvtx3::scoped_range r("Pipeline");    
        if (load_matrix_flag){
            cout << "> Loading matrices from file: " << input_file_name << endl;
            nvtx3::scoped_range r("Load Matrices");
            load_matrices(input_file_name, m1, m2, m1_rows, m1_cols, m2_cols);
        }
        else {
            // Create matrices
            cout << "> Creating random matrices." << endl;
            nvtx3::scoped_range r("Creating Matrices");
            m1 = create_random_matrix(m1_rows, m1_cols);
            m2 = create_random_matrix(m1_cols, m2_cols);
        }
        // print running parameters
        cout << ">> size: M1["<< m1_rows << ", " << m1_cols << "] M2[" << m1_cols << ", " << m2_cols << "]" << endl;
        if (save_matrix_flag){
            cout << "> Saving matrices to file: " << output_file_name << endl;
            nvtx3::scoped_range r("Save Matrices");
            save_matrices(output_file_name, m1, m2, m1_rows, m1_cols, m2_cols);
        }
        // Run the CPU version
        cout << "> Running on CPU: " << (run_cpu_flag ? "yes" : "no") << endl;
        if (run_cpu_flag) {
            cpu_result = cpu_naive_multiplication(m1, m2, m1_rows, m1_cols, m2_cols);
            if (save_cpu_result_flag){
                cout << ">> Saving CPU result to file: " << cpu_result_file << endl;
                nvtx3::scoped_range r("Save CPU Result");
                save_matrix(cpu_result_file, cpu_result, m1_rows, m2_cols);
            }
        }
        // Run the OpenMP version
        cout << "> Running on OpenMP: " << (run_openmp_flag ? "yes" : "no") << endl;
        if (run_openmp_flag) {
            openmp_result = openmp_multiplication(m1, m2, m1_rows, m1_cols, m2_cols);
            if (save_openmp_result_flag){
                cout << ">> Saving OpenMP result to file: " << openmp_result_file << endl;
                nvtx3::scoped_range r("Save OpenMP Result");
                save_matrix(openmp_result_file, openmp_result, m1_rows, m2_cols);
            }
        }
        // Run the CUDA version
        cout << "> Running on CUDA: " << (run_cuda_flag ? "yes" : "no") << endl;
        if (run_cuda_flag) {
            try {
                cuda_result = cuda_multiplication(m1, m2, m1_rows, m1_cols, m2_cols);
            }
            catch (const std::runtime_error &e){
                cout << "! CUDA error: " << e.what() << endl;
                exit(1);
            }
            if (save_cuda_result_flag){
                cout << ">> Saving CUDA result to file: " << cuda_result_file << endl;
                nvtx3::scoped_range r("Save CUDA Result");
                save_matrix(cuda_result_file, cuda_result, m1_rows, m2_cols);
            }
        }
        // Run the CUDA block version
        cout << "> Running on CUDA block: " << (run_cudablock_flag ? "yes" : "no") << endl;
        if (run_cudablock_flag) {
            try {
                cudablock_result = cuda_block_multiplication(m1, m2, m1_rows, m1_cols, m2_cols, cuda_block_size);
            }
            catch (const std::runtime_error &e){
                cout << "! CUDA error: " << e.what() << endl;
                exit(1);
            }
            if (save_cudablock_result_flag){
                cout << ">> Saving CUDA block result to file: " << cudablock_result_file << endl;
                nvtx3::scoped_range r("Save CUDA Block Result");
                save_matrix(cudablock_result_file, cudablock_result, m1_rows, m2_cols);
            }
        }
        if (run_compare_flag){
            nvtx3::scoped_range r("Compare Results");
            if (cpu_result.size() == openmp_result.size()){
                cout << "> Comparing CPU and OpenMP results." << endl;
                float max_diff = max_norm(cpu_result, openmp_result);
                cout << ">> max difference(abs): " << max_diff << endl;
            }
            else {
                cout << "! CPU and OpenMP results have different sizes." << endl;
            }
            if (cpu_result.size() == cuda_result.size()){
                cout << "> Comparing CPU and CUDA results." << endl;
                float max_diff = max_norm(cpu_result, cuda_result);
                cout << ">> max difference(abs): " << max_diff << endl;
            }
            else {
                cout << "! CPU and CUDA results have different sizes." << endl;
            }
            if (openmp_result.size() == cuda_result.size()){
                cout << "> Comparing OpenMP and CUDA results." << endl;
                float max_diff = max_norm(openmp_result, cuda_result);
                cout << ">> max difference(abs): " << max_diff << endl;
            }
            else {
                cout << "! OpenMP and CUDA results have different sizes." << endl;
            }
            if (cpu_result.size() == cudablock_result.size()){
                cout << "> Comparing CPU and CUDA block results." << endl;
                float max_diff = max_norm(cpu_result, cudablock_result);
                cout << ">> max difference(abs): " << max_diff << endl;
            }
            else {
                cout << "! CPU and CUDA block results have different sizes." << endl;
            }

        }
        if (print_matrix_flag){
            cout << "> Printing Matrices" << endl;
            nvtx3::scoped_range r("Print Matrices");
            cout << ">> Matrix 1:" << endl;
            print_matrix(m1, m1_rows);
            cout << ">> Matrix 2:" << endl;
            print_matrix(m2, m1_cols);
            if (run_cpu_flag){
                cout << endl << ">> CPU result:" << endl;
                print_matrix(cpu_result, m1_rows);
            }
            if (run_openmp_flag){
                cout << endl << ">> OpenMP result:" << endl;
                print_matrix(openmp_result, m1_rows);
            }
            if (run_cuda_flag){
                cout << endl << ">> CUDA result:" << endl;
                print_matrix(cuda_result, m1_rows);
            }
        }
    }
    timer.stop();
    util::timers.flush();
    return 0;
}