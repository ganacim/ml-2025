#ifndef _GPU_ERROR_H_
#define _GPU_ERROR_H_

#include <string>

namespace util
{

void check_cuda_error(const std::string &msg="");

}


#endif
