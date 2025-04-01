cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

#! Change this setting to your project's name
# Set project name. 
set(CMAKE_CUDA_ARCHITECTURES 50)
project(ml_examples)

# Enable CUDA as a language
enable_language(CXX)
enable_language(CUDA)

# Check if CUDA is enabled
include(CheckLanguage)
check_language(CUDA)

# Enable OpenMP
find_package(OpenMP REQUIRED)

# output path for project's binaries
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# path for additional cmake modules, if needed
# set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/cmake")

# #Configure the C++/C compiler
# if( CMAKE_SIZEOF_VOID_P MATCHES 4 ) # 32-bit O.S.
#     set(BASE_FLAGS "-std=c++11 -Wno-deprecated -march=native -mfpmath=sse -fopenmp -fPIC -Wno-unknown-pragmas -Wno-unused-local-typedefs")
# else( CMAKE_SIZEOF_VOID_P MATCHES 4 ) # 64-bit O.S.
#     set(BASE_FLAGS "-std=c++11 -Wno-deprecated -m64 -fopenmp -fPIC -Wno-unknown-pragmas -Wno-unused-local-typedefs")
#     #set(BASE_FLAGS "-std=c++11 -Wno-deprecated -m64 -march=native -mfpmath=sse -fopenmp")
# endif( CMAKE_SIZEOF_VOID_P MATCHES 4 )

# set(RELEASE_FLAGS "-O4 -DNDEBUG -W -Wunused -Wall -pedantic -fomit-frame-pointer -ffast-math -flto")
# set(DEBUG_FLAGS "-O0 -ggdb")
# if( CMAKE_BUILD_TYPE EQUAL Debug )
#     #set(CMAKE_CXX_LINK_FLAGS "-pg")
#   set(CMAKE_CXX_LINK_FLAGS "")
# endif( CMAKE_BUILD_TYPE EQUAL Debug )

# set(CMAKE_CXX_FLAGS_RELEASE "${BASE_FLAGS} ${RELEASE_FLAGS} ")
# set(CMAKE_CXX_FLAGS_DEBUG "${BASE_FLAGS} ${DEBUG_FLAGS} ")
# set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${BASE_FLAGS} ${DEBUG_FLAGS} ${RELEASE_FLAGS}")
# set(CMAKE_CXX_FLAGS "${BASE_FLAGS}")

# set(CMAKE_C_FLAGS_RELEASE "${BASE_FLAGS} ${RELEASE_FLAGS} ")
# set(CMAKE_C_FLAGS_DEBUG "${BASE_FLAGS} ${DEBUG_FLAGS} ")
# set(CMAKE_C_FLAGS_RELWITHDEBINFO "${BASE_FLAGS} ${DEBUG_FLAGS} ${RELEASE_FLAGS}")
# set(CMAKE_C_FLAGS "${BASE_FLAGS}")

# if true prints compiler messagens
set(CMAKE_VERBOSE_MAKEFILE false )

# # Find additional libs for the project
# find_package(Base64)

# # Add folders to C/C++ include path
# set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR})
# include_directories(${INCLUDE_DIR})

# Adds folder to the project
add_subdirectory(01_cuda)

