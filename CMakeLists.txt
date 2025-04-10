cmake_minimum_required(VERSION 3.18)
project(ParallelEdgeDetection VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xlinker --as-needed")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CXX_STANDARD 17)
enable_language(CUDA)

# --- Sources ---
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(RESOURCES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/res/")

set(TARGETS
    sequential
    mpi
    omp
    cuda
)

# --- Add executables ---
add_executable(sequential "${SRC_DIR}/seq_main.cpp")
add_executable(mpi        "${SRC_DIR}/mpi_main.cpp")
add_executable(omp        "${SRC_DIR}/omp_main.cpp")
add_executable(cuda       "${SRC_DIR}/cuda_main.cu")

# --- Compile options ---
set(COMMON_DEBUG_FLAGS "-W" "-Wall" "-Wextra" "-g" "-Og" "-DDebug" "-pedantic" "-pedantic-errors")
set(COMMON_RELEASE_FLAGS "-DNDEBUG" "-O2")

foreach(tgt IN LISTS TARGETS)
    target_include_directories(${tgt} PUBLIC "${INCLUDE_DIR}")
    target_compile_options(${tgt} PRIVATE "-I${INCLUDE_DIR}")

    target_compile_definitions(${tgt} PUBLIC RESOURCES_PATH="${RESOURCES_DIR}")
    target_compile_definitions(${tgt} PUBLIC SEQUENTIAL_OUTPUT_PATH="${OUTPUT_DIR}/sequential_output/")

    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        target_compile_options(${tgt} PUBLIC ${COMMON_DEBUG_FLAGS})
    elseif(CMAKE_BUILD_TYPE MATCHES "Release")
        target_compile_options(${tgt} PUBLIC ${COMMON_RELEASE_FLAGS})
    endif()
endforeach()

# --- Target-specific additions ---
target_compile_definitions(mpi  PUBLIC PARALLEL_OUTPUT_PATH="${OUTPUT_DIR}/parallel_output/")
target_compile_definitions(omp  PUBLIC OMP_OUTPUT_PATH="${OUTPUT_DIR}/omp_output/")
target_compile_definitions(cuda PUBLIC CUDA_OUTPUT_PATH="${OUTPUT_DIR}/cuda_output/")

# Silence CUDA warnings
if(CMAKE_BUILD_TYPE MATCHES "Release")
    target_compile_options(cuda PUBLIC -w)
endif()

# --- MPI ---
find_package(MPI REQUIRED)
target_link_libraries(mpi PUBLIC MPI::MPI_CXX)

# --- OpenMP ---
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(omp PUBLIC OpenMP::OpenMP_CXX)
endif()

# --- CUDA specific ---
set_target_properties(cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --- Build messages ---
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "
    INPUT: Name of input image file
    OUTPUT: Name of output image file
    THREAD_COUNT: Number of cores/threads to allocate to the program
    SP_THREADS_PER_BLK: Number of SP threads to allocate to each SP Block (CUDA Only)
    ALTERNATE_SEQUENTIAL_OUTPUT: Name of the output image file from the sequential program
")
message(STATUS "Run SEQUENTIAL: SEQUENTIAL_EXECUTABLE <INPUT> <OUTPUT>")
message(STATUS "Run MPI: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} MPI_EXECUTABLE ${MPIEXEC_POSTFLAGS} <INPUT> <OUTPUT> <ALTERNATE_SEQUENTIAL_OUTPUT>")
message(STATUS "Run OMP: OMP_EXECUTABLE <INPUT> <OUTPUT> <THREAD_COUNT> <ALTERNATE_SEQUENTIAL_OUTPUT>")
message(STATUS "Run CUDA: CUDA_EXECUTABLE <INPUT> <OUTPUT> <SP_THREADS_PER_BLK> <ALTERNATE_SEQUENTIAL_OUTPUT>")
