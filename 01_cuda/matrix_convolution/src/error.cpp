#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>

#include "error.h"

namespace util
{

void check_cuda_error(const std::string &msg)
{
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        if(msg.empty())
            throw std::runtime_error(cudaGetErrorString(err));
        else
        {
            std::stringstream ss;
            ss << msg << ": " << cudaGetErrorString(err);
            throw std::runtime_error(ss.str().c_str());
        }
    }
}

} // namespace util
