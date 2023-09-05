cmake_minimum_required(VERSION 3.20.0)
project(SimpleProject)
project(assembler C ASM)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

add_compile_options(-fno-rtti -fno-exceptions -fno-omit-frame-pointer)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Set your project compile flags.
# E.g. if using the C++ header files
# you will need to enable C++11 support
# for your compiler.

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

add_executable(simple-tool tool.cpp small_gemm.s)

# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs support core)

# FIX IMPORT PATH
add_library(libtpp_c_runner_utils SHARED IMPORTED)
set_property(TARGET libtpp_c_runner_utils 
  PROPERTY IMPORTED_LOCATION "/home/lorenzo/tpp-sandbox/build/lib/libtpp_c_runner_utils.so")

# Link against LLVM libraries
target_link_libraries(simple-tool PRIVATE ${llvm_libs})
# Link against MLIR libraries
target_link_libraries(simple-tool PRIVATE
  MLIRIR
  MLIRSupport
  libtpp_c_runner_utils
  )

option(BENCHMARK_DOWNLOAD_DEPENDENCIES 
  "Allow the downloading and in-tree building of unmet dependencies" ON)

add_subdirectory(benchmark)
target_link_libraries(simple-tool PRIVATE benchmark::benchmark)
