cmake_minimum_required(VERSION 3.9)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(342pr2 VERSION 0.1.0 LANGUAGES CXX CUDA)

add_executable(sequential "${CMAKE_CURRENT_SOURCE_DIR}/src/seq_main.cpp")
add_executable(omp "${CMAKE_CURRENT_SOURCE_DIR}/src/omp_main.cpp")
add_executable(cuda "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_main.cu")

find_package(MPI REQUIRED)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message("BUILDING DEBUG!")
    target_compile_options(sequential
            PUBLIC -W -Wall -Wextra -g -Og -DDebug -pedantic -pedantic-errors)
    target_compile_options(omp
            PUBLIC -W -Wall -Wextra -g -Og -DDebug -pedantic -pedantic-errors)
endif()

if(CMAKE_BUILD_TYPE MATCHES "Release")
    message("BUILDING RELEASE!")
    target_compile_options(sequential
            PUBLIC -DNDEBUG -O2)
    target_compile_options(omp
            PUBLIC -DNDEBUG -O2)
    # Silence CUDA Warnings
    target_compile_options(cuda
            PUBLIC -w)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(omp PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(sequential PUBLIC MPI::MPI_CXX)
target_link_libraries(omp PUBLIC MPI::MPI_CXX)
target_link_libraries(cuda PUBLIC MPI::MPI_CXX)

target_compile_definitions(sequential PUBLIC RESOURCES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/res/")
target_compile_definitions(omp PUBLIC RESOURCES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/res/")
target_compile_definitions(cuda PUBLIC RESOURCES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/res/")

target_compile_definitions(sequential PUBLIC SEQUENTIAL_OUTPUT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/sequential_output/")
target_compile_definitions(omp PUBLIC SEQUENTIAL_OUTPUT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/sequential_output/")
target_compile_definitions(omp PUBLIC OMP_OUTPUT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/omp_output/")
target_compile_definitions(cuda PUBLIC SEQUENTIAL_OUTPUT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/sequential_output/")
target_compile_definitions(cuda PUBLIC CUDA_OUTPUT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/cuda_output/")

target_include_directories(sequential PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_include_directories(omp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_include_directories(cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")

set_target_properties(cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
