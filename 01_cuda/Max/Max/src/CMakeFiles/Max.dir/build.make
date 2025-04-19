# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mahmut/ml-2025/01_cuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mahmut/ml-2025/01_cuda/Max

# Include any dependencies generated for this target.
include Max/src/CMakeFiles/Max.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include Max/src/CMakeFiles/Max.dir/compiler_depend.make

# Include the progress variables for this target.
include Max/src/CMakeFiles/Max.dir/progress.make

# Include the compile flags for this target's objects.
include Max/src/CMakeFiles/Max.dir/flags.make

Max/src/CMakeFiles/Max.dir/main.cpp.o: Max/src/CMakeFiles/Max.dir/flags.make
Max/src/CMakeFiles/Max.dir/main.cpp.o: src/main.cpp
Max/src/CMakeFiles/Max.dir/main.cpp.o: Max/src/CMakeFiles/Max.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/mahmut/ml-2025/01_cuda/Max/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Max/src/CMakeFiles/Max.dir/main.cpp.o"
	cd /home/mahmut/ml-2025/01_cuda/Max/Max/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Max/src/CMakeFiles/Max.dir/main.cpp.o -MF CMakeFiles/Max.dir/main.cpp.o.d -o CMakeFiles/Max.dir/main.cpp.o -c /home/mahmut/ml-2025/01_cuda/Max/src/main.cpp

Max/src/CMakeFiles/Max.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Max.dir/main.cpp.i"
	cd /home/mahmut/ml-2025/01_cuda/Max/Max/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mahmut/ml-2025/01_cuda/Max/src/main.cpp > CMakeFiles/Max.dir/main.cpp.i

Max/src/CMakeFiles/Max.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Max.dir/main.cpp.s"
	cd /home/mahmut/ml-2025/01_cuda/Max/Max/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mahmut/ml-2025/01_cuda/Max/src/main.cpp -o CMakeFiles/Max.dir/main.cpp.s

Max/src/CMakeFiles/Max.dir/kernel.cu.o: Max/src/CMakeFiles/Max.dir/flags.make
Max/src/CMakeFiles/Max.dir/kernel.cu.o: src/kernel.cu
Max/src/CMakeFiles/Max.dir/kernel.cu.o: Max/src/CMakeFiles/Max.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/mahmut/ml-2025/01_cuda/Max/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object Max/src/CMakeFiles/Max.dir/kernel.cu.o"
	cd /home/mahmut/ml-2025/01_cuda/Max/Max/src && /usr/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT Max/src/CMakeFiles/Max.dir/kernel.cu.o -MF CMakeFiles/Max.dir/kernel.cu.o.d -x cu -c /home/mahmut/ml-2025/01_cuda/Max/src/kernel.cu -o CMakeFiles/Max.dir/kernel.cu.o

Max/src/CMakeFiles/Max.dir/kernel.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/Max.dir/kernel.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

Max/src/CMakeFiles/Max.dir/kernel.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/Max.dir/kernel.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

Max/src/CMakeFiles/Max.dir/timer.cpp.o: Max/src/CMakeFiles/Max.dir/flags.make
Max/src/CMakeFiles/Max.dir/timer.cpp.o: src/timer.cpp
Max/src/CMakeFiles/Max.dir/timer.cpp.o: Max/src/CMakeFiles/Max.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/mahmut/ml-2025/01_cuda/Max/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object Max/src/CMakeFiles/Max.dir/timer.cpp.o"
	cd /home/mahmut/ml-2025/01_cuda/Max/Max/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Max/src/CMakeFiles/Max.dir/timer.cpp.o -MF CMakeFiles/Max.dir/timer.cpp.o.d -o CMakeFiles/Max.dir/timer.cpp.o -c /home/mahmut/ml-2025/01_cuda/Max/src/timer.cpp

Max/src/CMakeFiles/Max.dir/timer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Max.dir/timer.cpp.i"
	cd /home/mahmut/ml-2025/01_cuda/Max/Max/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mahmut/ml-2025/01_cuda/Max/src/timer.cpp > CMakeFiles/Max.dir/timer.cpp.i

Max/src/CMakeFiles/Max.dir/timer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Max.dir/timer.cpp.s"
	cd /home/mahmut/ml-2025/01_cuda/Max/Max/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mahmut/ml-2025/01_cuda/Max/src/timer.cpp -o CMakeFiles/Max.dir/timer.cpp.s

Max/src/CMakeFiles/Max.dir/Vector.cpp.o: Max/src/CMakeFiles/Max.dir/flags.make
Max/src/CMakeFiles/Max.dir/Vector.cpp.o: src/Vector.cpp
Max/src/CMakeFiles/Max.dir/Vector.cpp.o: Max/src/CMakeFiles/Max.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/mahmut/ml-2025/01_cuda/Max/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object Max/src/CMakeFiles/Max.dir/Vector.cpp.o"
	cd /home/mahmut/ml-2025/01_cuda/Max/Max/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Max/src/CMakeFiles/Max.dir/Vector.cpp.o -MF CMakeFiles/Max.dir/Vector.cpp.o.d -o CMakeFiles/Max.dir/Vector.cpp.o -c /home/mahmut/ml-2025/01_cuda/Max/src/Vector.cpp

Max/src/CMakeFiles/Max.dir/Vector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Max.dir/Vector.cpp.i"
	cd /home/mahmut/ml-2025/01_cuda/Max/Max/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mahmut/ml-2025/01_cuda/Max/src/Vector.cpp > CMakeFiles/Max.dir/Vector.cpp.i

Max/src/CMakeFiles/Max.dir/Vector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Max.dir/Vector.cpp.s"
	cd /home/mahmut/ml-2025/01_cuda/Max/Max/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mahmut/ml-2025/01_cuda/Max/src/Vector.cpp -o CMakeFiles/Max.dir/Vector.cpp.s

Max/src/CMakeFiles/Max.dir/error.cpp.o: Max/src/CMakeFiles/Max.dir/flags.make
Max/src/CMakeFiles/Max.dir/error.cpp.o: src/error.cpp
Max/src/CMakeFiles/Max.dir/error.cpp.o: Max/src/CMakeFiles/Max.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/mahmut/ml-2025/01_cuda/Max/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object Max/src/CMakeFiles/Max.dir/error.cpp.o"
	cd /home/mahmut/ml-2025/01_cuda/Max/Max/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Max/src/CMakeFiles/Max.dir/error.cpp.o -MF CMakeFiles/Max.dir/error.cpp.o.d -o CMakeFiles/Max.dir/error.cpp.o -c /home/mahmut/ml-2025/01_cuda/Max/src/error.cpp

Max/src/CMakeFiles/Max.dir/error.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Max.dir/error.cpp.i"
	cd /home/mahmut/ml-2025/01_cuda/Max/Max/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mahmut/ml-2025/01_cuda/Max/src/error.cpp > CMakeFiles/Max.dir/error.cpp.i

Max/src/CMakeFiles/Max.dir/error.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Max.dir/error.cpp.s"
	cd /home/mahmut/ml-2025/01_cuda/Max/Max/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mahmut/ml-2025/01_cuda/Max/src/error.cpp -o CMakeFiles/Max.dir/error.cpp.s

Max/src/CMakeFiles/Max.dir/cpu.cpp.o: Max/src/CMakeFiles/Max.dir/flags.make
Max/src/CMakeFiles/Max.dir/cpu.cpp.o: src/cpu.cpp
Max/src/CMakeFiles/Max.dir/cpu.cpp.o: Max/src/CMakeFiles/Max.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/mahmut/ml-2025/01_cuda/Max/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object Max/src/CMakeFiles/Max.dir/cpu.cpp.o"
	cd /home/mahmut/ml-2025/01_cuda/Max/Max/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Max/src/CMakeFiles/Max.dir/cpu.cpp.o -MF CMakeFiles/Max.dir/cpu.cpp.o.d -o CMakeFiles/Max.dir/cpu.cpp.o -c /home/mahmut/ml-2025/01_cuda/Max/src/cpu.cpp

Max/src/CMakeFiles/Max.dir/cpu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Max.dir/cpu.cpp.i"
	cd /home/mahmut/ml-2025/01_cuda/Max/Max/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mahmut/ml-2025/01_cuda/Max/src/cpu.cpp > CMakeFiles/Max.dir/cpu.cpp.i

Max/src/CMakeFiles/Max.dir/cpu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Max.dir/cpu.cpp.s"
	cd /home/mahmut/ml-2025/01_cuda/Max/Max/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mahmut/ml-2025/01_cuda/Max/src/cpu.cpp -o CMakeFiles/Max.dir/cpu.cpp.s

Max/src/CMakeFiles/Max.dir/cuda.cpp.o: Max/src/CMakeFiles/Max.dir/flags.make
Max/src/CMakeFiles/Max.dir/cuda.cpp.o: src/cuda.cpp
Max/src/CMakeFiles/Max.dir/cuda.cpp.o: Max/src/CMakeFiles/Max.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/mahmut/ml-2025/01_cuda/Max/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object Max/src/CMakeFiles/Max.dir/cuda.cpp.o"
	cd /home/mahmut/ml-2025/01_cuda/Max/Max/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Max/src/CMakeFiles/Max.dir/cuda.cpp.o -MF CMakeFiles/Max.dir/cuda.cpp.o.d -o CMakeFiles/Max.dir/cuda.cpp.o -c /home/mahmut/ml-2025/01_cuda/Max/src/cuda.cpp

Max/src/CMakeFiles/Max.dir/cuda.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Max.dir/cuda.cpp.i"
	cd /home/mahmut/ml-2025/01_cuda/Max/Max/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mahmut/ml-2025/01_cuda/Max/src/cuda.cpp > CMakeFiles/Max.dir/cuda.cpp.i

Max/src/CMakeFiles/Max.dir/cuda.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Max.dir/cuda.cpp.s"
	cd /home/mahmut/ml-2025/01_cuda/Max/Max/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mahmut/ml-2025/01_cuda/Max/src/cuda.cpp -o CMakeFiles/Max.dir/cuda.cpp.s

# Object files for target Max
Max_OBJECTS = \
"CMakeFiles/Max.dir/main.cpp.o" \
"CMakeFiles/Max.dir/kernel.cu.o" \
"CMakeFiles/Max.dir/timer.cpp.o" \
"CMakeFiles/Max.dir/Vector.cpp.o" \
"CMakeFiles/Max.dir/error.cpp.o" \
"CMakeFiles/Max.dir/cpu.cpp.o" \
"CMakeFiles/Max.dir/cuda.cpp.o"

# External object files for target Max
Max_EXTERNAL_OBJECTS =

Max/src/Max: Max/src/CMakeFiles/Max.dir/main.cpp.o
Max/src/Max: Max/src/CMakeFiles/Max.dir/kernel.cu.o
Max/src/Max: Max/src/CMakeFiles/Max.dir/timer.cpp.o
Max/src/Max: Max/src/CMakeFiles/Max.dir/Vector.cpp.o
Max/src/Max: Max/src/CMakeFiles/Max.dir/error.cpp.o
Max/src/Max: Max/src/CMakeFiles/Max.dir/cpu.cpp.o
Max/src/Max: Max/src/CMakeFiles/Max.dir/cuda.cpp.o
Max/src/Max: Max/src/CMakeFiles/Max.dir/build.make
Max/src/Max: /usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so
Max/src/Max: /usr/lib/x86_64-linux-gnu/libpthread.a
Max/src/Max: Max/src/CMakeFiles/Max.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/mahmut/ml-2025/01_cuda/Max/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX executable Max"
	cd /home/mahmut/ml-2025/01_cuda/Max/Max/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Max.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Max/src/CMakeFiles/Max.dir/build: Max/src/Max
.PHONY : Max/src/CMakeFiles/Max.dir/build

Max/src/CMakeFiles/Max.dir/clean:
	cd /home/mahmut/ml-2025/01_cuda/Max/Max/src && $(CMAKE_COMMAND) -P CMakeFiles/Max.dir/cmake_clean.cmake
.PHONY : Max/src/CMakeFiles/Max.dir/clean

Max/src/CMakeFiles/Max.dir/depend:
	cd /home/mahmut/ml-2025/01_cuda/Max && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mahmut/ml-2025/01_cuda /home/mahmut/ml-2025/01_cuda/Max/src /home/mahmut/ml-2025/01_cuda/Max /home/mahmut/ml-2025/01_cuda/Max/Max/src /home/mahmut/ml-2025/01_cuda/Max/Max/src/CMakeFiles/Max.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : Max/src/CMakeFiles/Max.dir/depend

