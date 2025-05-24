cmake_minimum_required(VERSION 3.18)
project(ntxent_cuda 
    LANGUAGES CXX CUDA 
    VERSION 1.0.0
    DESCRIPTION "CUDA-accelerated NT-Xent loss implementation"
)

# Options
option(BUILD_TESTS "Build tests" ON)
option(ENABLE_PROFILING "Enable CUDA profiling" OFF)
option(USE_TENSOR_CORES "Enable Tensor Cores (requires Volta+ GPU)" ON)
option(BUILD_PYTHON_MODULE "Build Python bindings" ON)
option(ENABLE_MPI "Enable MPI support" OFF)
option(ENABLE_NCCL "Enable NCCL support" OFF)
option(USE_FP16 "Enable FP16 support" ON)
option(BUILD_BENCHMARKS "Build benchmark suite" ON)

# Version requirements
set(MIN_CUDA_VERSION 11.0)
set(MIN_PYTORCH_VERSION 1.8.0)
set(MIN_CMAKE_VERSION 3.18)

# Check versions
if(CMAKE_VERSION VERSION_LESS ${MIN_CMAKE_VERSION})
    message(FATAL_ERROR "CMake version ${CMAKE_VERSION} is too old. Minimum required is ${MIN_CMAKE_VERSION}")
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS ${MIN_CUDA_VERSION})
    message(FATAL_ERROR "CUDA version ${CMAKE_CUDA_COMPILER_VERSION} is too old. Minimum required is ${MIN_CUDA_VERSION}")
endif()

# Find required packages
find_package(Torch REQUIRED)
if(Torch_VERSION VERSION_LESS ${MIN_PYTORCH_VERSION})
    message(FATAL_ERROR "PyTorch version ${Torch_VERSION} is too old. Minimum required is ${MIN_PYTORCH_VERSION}")
endif()

find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)

if(ENABLE_MPI)
    find_package(MPI REQUIRED)
endif()

if(ENABLE_NCCL)
    find_package(NCCL REQUIRED)
endif()

# Set standards and flags
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Detect CUDA architectures
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(CUDA_ARCHS)
string(STRIP "${CUDA_ARCHS}" CUDA_ARCHS)
string(REPLACE " " ";" CUDA_ARCHS "${CUDA_ARCHS}")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHS})

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -O3 \
    -use_fast_math \
    --expt-relaxed-constexpr \
    --expt-extended-lambda \
    --generate-line-info \
    --ptxas-options=-v")

if(USE_TENSOR_CORES)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
        -DUSE_TENSOR_CORES \
        --gpu-architecture=compute_70")
endif()

if(USE_FP16)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_FP16")
endif()

if(ENABLE_PROFILING)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -DENABLE_PROFILING")
endif()

# Add compiler warnings
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /W4")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra,-Werror")
endif()

# Library target
add_library(ntxent_cuda SHARED
    src/ntxent_kernel.cu
    src/binding.cpp)

target_include_directories(ntxent_cuda 
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDA_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(ntxent_cuda 
    PRIVATE 
        ${TORCH_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        pybind11::module
)

if(ENABLE_MPI)
    target_link_libraries(ntxent_cuda PRIVATE MPI::MPI_CXX)
endif()

if(ENABLE_NCCL)
    target_link_libraries(ntxent_cuda PRIVATE ${NCCL_LIBRARIES})
endif()

# Benchmark executable
if(BUILD_BENCHMARKS)
    add_executable(benchmark src/benchmark.cpp)
    target_link_libraries(benchmark PRIVATE ntxent_cuda)
endif()

# Testing
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(tests)
endif()

# Installation
include(GNUInstallDirs)
install(TARGETS ntxent_cuda
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES 
    include/ntxent_kernel.cuh
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ntxent
)
