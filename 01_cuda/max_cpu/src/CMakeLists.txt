add_executable(max_cpu
    main.cpp
    kernel.cu kernel.h
    timer.cpp timer.h
    error.cpp error.h
)

# if using add_library
# target_compile_features(template PUBLIC cxx_std_11)
# set_target_properties(template PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
