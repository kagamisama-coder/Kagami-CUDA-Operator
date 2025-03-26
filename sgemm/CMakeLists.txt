cmake_minimum_required(VERSION 3.10)

project(KagamiOperator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

set(CMAKE_CUDA_ARCHITECTURES 75)

add_executable(sgemm sgemm.cu utils.cu)

set_source_files_properties(sgemm.cu utils.cu PROPERTIES LANGUAGE CUDA)

target_link_libraries(sgemm ${CUDA_LIBRARIES} cublas)

target_include_directories(sgemm PRIVATE ${PROJECT_SOURCE_DIR}/src ${CUDA_INCLUDE_DIRS})

target_link_directories(sgemm PRIVATE ${CUDA_LIBRARY_DIRS})


