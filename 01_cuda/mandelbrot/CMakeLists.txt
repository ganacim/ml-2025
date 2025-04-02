find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

add_executable(mandelbrot_GPU
    main.cpp
    kernel.cu kernel.h
)
target_link_libraries(mandelbrot_GPU OpenGL::GL GLUT::GLUT GLU)

add_executable(mandelbrot_CPU
    main_cpu.cpp
)
target_link_libraries(mandelbrot_CPU OpenGL::GL GLUT::GLUT GLU)
# if using add_library
# target_compile_features(template PUBLIC cxx_std_11)
# set_target_properties(template PROPERTIES CUDA_SEPARABLE_COMPILATION ON)