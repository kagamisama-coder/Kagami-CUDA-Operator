#pragma once
#include <cublas_v2.h>

#include <iostream>

#define CEIL(a, b) ((a + b - 1) / (b))
#define CUDA_CHECK(err) cudaCheck((err), __FILE__, __LINE__)

void cudaCheck(cudaError_t err, const char *file, int line);

bool verify_matrix(float *matrix1, float *matrix2, int M, int N);

void print_matrix(float *matrix, int M, int N);

void CudaDeviceInfo();

void init_matrix(float *matrix, int M, int N);

void print_kernel_info(int kernel_num, int milliseconds, int M, int N, int K);

void write_kernel_info(int kernel_num, float milliseconds, int M, int N, int K);

void test_cublas(cublasHandle_t handle, int M, int N, int K, float alpha,
                 float *A, float *B, float beta, float *C);

void test_kernel1(float *A, float *B, float *C, int M, int N, int K);

void test_kernel2(float *A, float *B, float *C, int M, int N, int K);

void test_kernel3(float *A, float *B, float *C, int M, int N, int K);

void test_kernel4(float *A, float *B, float *C, int M, int N, int K);

void test_kernel5(float *A, float *B, float *C, int M, int N, int K);

void test_kernel6(float *A, float *B, float *C, int M, int N, int K);

void test_kernel(int kernel_num, float *A, float *B, float *C, int M, int N,
                 int K, cublasHandle_t handle);

void elapsed_time();