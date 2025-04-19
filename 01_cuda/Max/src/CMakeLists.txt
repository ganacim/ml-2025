cmake_minimum_required(VERSION 3.18)
project(Max LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 61)

find_package(OpenMP REQUIRED)

add_executable(Max
    main.cpp
    kernel.cu kernel.h
    timer.cpp timer.h
    Vector.cpp Vector.h
    error.cpp error.h
    cpu.cpp cpu.h
    cuda.cu cuda.h)  

set_source_files_properties(kernel.cu PROPERTIES LANGUAGE CUDA)



target_link_libraries(Max PRIVATE nvToolsExt OpenMP::OpenMP_CXX)
target_compile_options(Max PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)


 