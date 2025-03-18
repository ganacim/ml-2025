add_executable(exemplo
    main.cpp
)

# link with OpenMP
target_link_libraries(exemplo PUBLIC OpenMP::OpenMP_CXX)

# if using add_library
# target_compile_features(template PUBLIC cxx_std_11)
# set_target_properties(template PROPERTIES CUDA_SEPARABLE_COMPILATION ON)