#ifndef _CPU_H_
#define _CPU_H_

#include <vector>

float cpu_naive_maxvalue(const std::vector<float>& v);

float openmp_maxvalue(const std::vector<float>& v);

#endif