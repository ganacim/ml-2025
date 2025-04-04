add_executable(template
    main.cpp
    kernel.cu kernel.h
)

# if using add_library
# target_compile_features(template PUBLIC cxx_std_11)
# set_target_properties(template PROPERTIES CUDA_SEPARABLE_COMPILATION ON)