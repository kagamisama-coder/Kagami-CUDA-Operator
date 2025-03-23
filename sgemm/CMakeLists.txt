cmake_minimum_required(VERSION 3.10)

project(KagamiOperator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11) # 设置 C++ 版本
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3") # 设置编译选项

# 设置可执行文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

# 指定 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 75) # 例如：针对 Turing 架构（如 RTX 2080）

# 添加可执行文件
add_executable(sgemm sgemm.cu utils.cu)

# 设置 CUDA 源文件的语言属性
set_source_files_properties(sgemm.cu utils.cu PROPERTIES LANGUAGE CUDA)

# 链接 CUDA 库
target_link_libraries(sgemm ${CUDA_LIBRARIES} cublas)

# 配置目标头文件搜索路径
target_include_directories(sgemm PRIVATE ${PROJECT_SOURCE_DIR}/src ${CUDA_INCLUDE_DIRS})

# 配置目标链接库目录
target_link_directories(sgemm PRIVATE ${CUDA_LIBRARY_DIRS})


