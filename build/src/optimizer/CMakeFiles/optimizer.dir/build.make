# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.29

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\CMake\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\pawel\OneDrive\Desktop\CLife

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\pawel\OneDrive\Desktop\CLife\build

# Include any dependencies generated for this target.
include src/optimizer/CMakeFiles/optimizer.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/optimizer/CMakeFiles/optimizer.dir/compiler_depend.make

# Include the progress variables for this target.
include src/optimizer/CMakeFiles/optimizer.dir/progress.make

# Include the compile flags for this target's objects.
include src/optimizer/CMakeFiles/optimizer.dir/flags.make

src/optimizer/CMakeFiles/optimizer.dir/src/NeuralNetworkOptimizer.cpp.obj: src/optimizer/CMakeFiles/optimizer.dir/flags.make
src/optimizer/CMakeFiles/optimizer.dir/src/NeuralNetworkOptimizer.cpp.obj: src/optimizer/CMakeFiles/optimizer.dir/includes_CXX.rsp
src/optimizer/CMakeFiles/optimizer.dir/src/NeuralNetworkOptimizer.cpp.obj: C:/Users/pawel/OneDrive/Desktop/CLife/src/optimizer/src/NeuralNetworkOptimizer.cpp
src/optimizer/CMakeFiles/optimizer.dir/src/NeuralNetworkOptimizer.cpp.obj: src/optimizer/CMakeFiles/optimizer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:\Users\pawel\OneDrive\Desktop\CLife\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/optimizer/CMakeFiles/optimizer.dir/src/NeuralNetworkOptimizer.cpp.obj"
	cd /d C:\Users\pawel\OneDrive\Desktop\CLife\build\src\optimizer && C:\msys64\ucrt64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/optimizer/CMakeFiles/optimizer.dir/src/NeuralNetworkOptimizer.cpp.obj -MF CMakeFiles\optimizer.dir\src\NeuralNetworkOptimizer.cpp.obj.d -o CMakeFiles\optimizer.dir\src\NeuralNetworkOptimizer.cpp.obj -c C:\Users\pawel\OneDrive\Desktop\CLife\src\optimizer\src\NeuralNetworkOptimizer.cpp

src/optimizer/CMakeFiles/optimizer.dir/src/NeuralNetworkOptimizer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/optimizer.dir/src/NeuralNetworkOptimizer.cpp.i"
	cd /d C:\Users\pawel\OneDrive\Desktop\CLife\build\src\optimizer && C:\msys64\ucrt64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\pawel\OneDrive\Desktop\CLife\src\optimizer\src\NeuralNetworkOptimizer.cpp > CMakeFiles\optimizer.dir\src\NeuralNetworkOptimizer.cpp.i

src/optimizer/CMakeFiles/optimizer.dir/src/NeuralNetworkOptimizer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/optimizer.dir/src/NeuralNetworkOptimizer.cpp.s"
	cd /d C:\Users\pawel\OneDrive\Desktop\CLife\build\src\optimizer && C:\msys64\ucrt64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\pawel\OneDrive\Desktop\CLife\src\optimizer\src\NeuralNetworkOptimizer.cpp -o CMakeFiles\optimizer.dir\src\NeuralNetworkOptimizer.cpp.s

# Object files for target optimizer
optimizer_OBJECTS = \
"CMakeFiles/optimizer.dir/src/NeuralNetworkOptimizer.cpp.obj"

# External object files for target optimizer
optimizer_EXTERNAL_OBJECTS =

src/optimizer/liboptimizer.a: src/optimizer/CMakeFiles/optimizer.dir/src/NeuralNetworkOptimizer.cpp.obj
src/optimizer/liboptimizer.a: src/optimizer/CMakeFiles/optimizer.dir/build.make
src/optimizer/liboptimizer.a: src/optimizer/CMakeFiles/optimizer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=C:\Users\pawel\OneDrive\Desktop\CLife\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library liboptimizer.a"
	cd /d C:\Users\pawel\OneDrive\Desktop\CLife\build\src\optimizer && $(CMAKE_COMMAND) -P CMakeFiles\optimizer.dir\cmake_clean_target.cmake
	cd /d C:\Users\pawel\OneDrive\Desktop\CLife\build\src\optimizer && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\optimizer.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/optimizer/CMakeFiles/optimizer.dir/build: src/optimizer/liboptimizer.a
.PHONY : src/optimizer/CMakeFiles/optimizer.dir/build

src/optimizer/CMakeFiles/optimizer.dir/clean:
	cd /d C:\Users\pawel\OneDrive\Desktop\CLife\build\src\optimizer && $(CMAKE_COMMAND) -P CMakeFiles\optimizer.dir\cmake_clean.cmake
.PHONY : src/optimizer/CMakeFiles/optimizer.dir/clean

src/optimizer/CMakeFiles/optimizer.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\pawel\OneDrive\Desktop\CLife C:\Users\pawel\OneDrive\Desktop\CLife\src\optimizer C:\Users\pawel\OneDrive\Desktop\CLife\build C:\Users\pawel\OneDrive\Desktop\CLife\build\src\optimizer C:\Users\pawel\OneDrive\Desktop\CLife\build\src\optimizer\CMakeFiles\optimizer.dir\DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/optimizer/CMakeFiles/optimizer.dir/depend
