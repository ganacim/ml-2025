#include <iostream>
#include <omp.h>

using namespace std;

void print() {
    cout << "Hello OpenMP World from thread " << omp_get_thread_num() << endl;
}

int main(int argc, const char* argv[]) {
    cout << "Hello OpenMP World!" << endl;

    omp_set_num_threads(4);
    #pragma omp parallel
    {
        print();
    }
    cerr << "End of program" << endl;
    return 0;
}