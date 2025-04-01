find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

add_executable(mandelbrot
    main.cpp
    kernel.cu kernel.h
)

target_link_libraries(template OpenGL::GL GLUT::GLUT GLU)

# if using add_library
# target_compile_features(template PUBLIC cxx_std_11)
# set_target_properties(template PROPERTIES CUDA_SEPARABLE_COMPILATION ON)