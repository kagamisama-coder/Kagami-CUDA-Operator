#include <cstdlib>
#include <iomanip>

#include "kernel.cuh"
#include "utils.cuh"

void cudaCheck(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    std::cout << "[ERROR]: CUDA error at file " << file << " line " << line
              << ", info: " << cudaGetErrorString(err) << std::endl;
  }
}

bool verify_matrix(float *matrix1, float *matrix2, int M, int N) {
  if (matrix1 == nullptr || matrix2 == nullptr) return false;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (std::fabs(matrix1[i * N + j] - matrix2[i * N + j]) > 1e-4) {
        std::cerr << "Matrix verification failed at element [" << i << "][" << j
                  << "]!" << std::endl;
        std::cerr << "matrix1[" << i << "][" << j
                  << "] = " << matrix1[i * N + j] << std::endl;
        std::cerr << "matrix2[" << i << "][" << j
                  << "] = " << matrix2[i * N + j] << std::endl;
        return false;
      }
    }
  }
  std::cout << "Matrix verification successful!" << std::endl;
  return true;
}

void print_matrix(float *matrix, int M, int N) {
  if (matrix == nullptr) {
    std::cerr << "Print matrix failed because matrix is null!" << std::endl;
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (j == N - 1)
        std::cout << matrix[i * N + j] << std::endl;
      else
        std::cout << matrix[i * N + j] << " ";
    }
  }
}

void CudaDeviceInfo() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0) {
    std::cerr << "No CUDA-capable devices found!" << std::endl;
    return;
  }

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);

    std::cout << "==============================" << std::endl;
    std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
    std::cout << "-----------------------------" << std::endl;
    std::cout << "  Compute Capability: " << deviceProp.major << "."
              << deviceProp.minor << std::endl;
    std::cout << "  Total Global Memory: "
              << deviceProp.totalGlobalMem / (1024 * 1024) << " MB"
              << std::endl;
    std::cout << "  Shared Memory per Block: "
              << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "  Warp Size: " << deviceProp.warpSize << std::endl;
    std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock
              << std::endl;
    std::cout << "  Max Threads Dimension: (" << deviceProp.maxThreadsDim[0]
              << ", " << deviceProp.maxThreadsDim[1] << ", "
              << deviceProp.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "  Max Grid Size: (" << deviceProp.maxGridSize[0] << ", "
              << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2]
              << ")" << std::endl;
    std::cout << "  Clock Rate: " << deviceProp.clockRate / 1000 << " MHz"
              << std::endl;
    std::cout << "  Number of Multiprocessors: "
              << deviceProp.multiProcessorCount << std::endl;
    std::cout << "  Integrated GPU (with host memory): "
              << (deviceProp.integrated ? "Yes" : "No") << std::endl;
    std::cout << "  Can Map Host Memory: "
              << (deviceProp.canMapHostMemory ? "Yes" : "No") << std::endl;
    std::cout << "  Concurrent Kernels: "
              << (deviceProp.concurrentKernels ? "Yes" : "No") << std::endl;
    std::cout << "  ECC Enabled: " << (deviceProp.ECCEnabled ? "Yes" : "No")
              << std::endl;
    std::cout << "==============================" << std::endl;
  }
}

void init_matrix(float *matrix, int M, int N) {
  for (int i = 0; i < M * N; i++) {
    matrix[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
}

void print_kernel_info(int kernel_num, int milliseconds, int M, int N, int K){
  float gflops = 1.0 * 2 * M * N * K * 1000 * 1e-9 / milliseconds;
  std::cout << "Kernel num: " << kernel_num << " Cost time: " << std::fixed
            << milliseconds / 1000 << "s" << " Performance: " << std::fixed
            << gflops << " GFLOPS" << std::endl;
}

void test_cublas(cublasHandle_t handle, int M, int N, int K, float alpha,
                 float *A, float *B, float beta, float *C) {
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K,
              &beta, C, N);
}

void test_kernel1(float *A, float *B, float *C, int M, int N, int K) {
  dim3 blockDim(32, 32);
  dim3 gridDim(CEIL(M, 32), CEIL(N, 32));
  sgemm_v1<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

void test_kernel2(float *A, float *B, float *C, int M, int N, int K) {
  dim3 blockDim(32 * 32);
  dim3 gridDim(CEIL(M, 32), CEIL(N, 32));
  sgemm_v2<32><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

void test_kernel3(float *A, float *B, float *C, int M, int N, int K){
  dim3 blockDim(64 * 8);
  dim3 gridDim(CEIL(M, 64), CEIL(N, 64));
  sgemm_v3<64, 64, 8, 8><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

// 一个block是线程是(16, 16), 每个线程负责(8, 8), block负责的矩阵区域是(128, 128) 
void test_kernel4(float *A, float *B, float *C, int M, int N, int K){
  dim3 blockDim(16 * 16);
  dim3 gridDim(CEIL(M, 128), CEIL(N, 128));
  sgemm_v4<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

void test_kernel5(float *A, float *B, float *C, int M, int N, int K){
  dim3 blockDim(16 * 16);
  dim3 gridDim(CEIL(M, 128), CEIL(N, 128));
  sgemm_v5<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

void test_kernel6(float *A, float *B, float *C, int M, int N, int K){
  dim3 blockDim(16 * 16);
  dim3 gridDim(CEIL(M, 128), CEIL(N, 128));
  sgemm_v6<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

void test_kernel(int kernel_num, float *A, float *B, float *C, int M, int N,
                 int K, cublasHandle_t handle) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  switch (kernel_num) {
    case 0:
      test_cublas(handle, M, N, K, 1.0, A, B, 0.0, C);
      break;
    case 1:
      test_kernel1(A, B, C, M, N, K);
      break;
    case 2:
      test_kernel2(A, B, C, M, N, K);
      break;
    case 3:
      test_kernel3(A, B, C, M, N, K);
      break;
    case 4:
      test_kernel4(A, B, C, M, N, K);
      break;
    case 5:
      test_kernel5(A, B, C, M, N, K);
      break;
    case 6:
      test_kernel6(A, B, C, M, N, K);
      break;
    default:
      std::cerr << "[ERROR]: Kernel num does not exist!" << std::endl;
      break;
  }

  cudaDeviceSynchronize();
  CUDA_CHECK(cudaEventRecord(stop));

  CUDA_CHECK(cudaEventSynchronize(stop));

  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  print_kernel_info(kernel_num, milliseconds, M, N, K);
}