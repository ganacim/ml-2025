add_executable(matrix_multiplication
    main.cpp
    cuda.cu cuda.h
    matrix.cpp matrix.h
    cpu.cpp cpu.h
    timer.cpp timer.h
    error.cpp error.h
)

target_link_libraries(matrix_multiplication PRIVATE nvtx3-cpp OpenMP::OpenMP_CXX)
target_compile_options(matrix_multiplication PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                                                    --generate-line-info
                                                    # --use_fast_math
                                                    # --relocatable-device-code=true
 >)

# if using add_library
# target_compile_features(template PUBLIC cxx_std_11)
# set_target_properties(template PROPERTIES CUDA_SEPARABLE_COMPILATION ON)