#include "utils.cuh"

int main() {
  const int M = 6144;
  const int N = 6144;
  const int K = 6144;
  CudaDeviceInfo();
  float *h_A = nullptr, *h_B = nullptr, *h_C = nullptr, *h_C_ref = nullptr;
  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_C_ref = nullptr;

  int size_A = M * K * sizeof(float);
  int size_B = K * N * sizeof(float);
  int size_C = M * N * sizeof(float);

  h_A = (float *)malloc(size_A);
  h_B = (float *)malloc(size_B);
  h_C = (float *)malloc(size_C);
  h_C_ref = (float *)malloc(size_C);

  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    printf("Create cublas handle error.\n");
    exit(EXIT_FAILURE);
  };

  CUDA_CHECK(cudaMalloc((void **)&d_A, size_A));
  CUDA_CHECK(cudaMalloc((void **)&d_B, size_B));
  CUDA_CHECK(cudaMalloc((void **)&d_C, size_C));
  CUDA_CHECK(cudaMalloc((void **)&d_C_ref, size_C));

  // warmup
  std::cout << "============Warm up===========" << std::endl;
  init_matrix(h_A, M, K);
  init_matrix(h_B, K, N);
  CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
  test_kernel(0, d_A, d_B, d_C, M, N, K, handle);
  cudaDeviceSynchronize();

  const int repeat_time = 10;

  for (int epoch = 0; epoch < repeat_time; epoch++) {
    std::cout << "============Epoch " << epoch + 1 << "===========" << std::endl;

    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    test_kernel(0, d_A, d_B, d_C_ref, M, N, K, handle);

    cudaMemcpy(h_C_ref, d_C_ref, size_C, cudaMemcpyDeviceToHost);

    test_kernel(3, d_A, d_B, d_C, M, N, K, handle);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    verify_matrix(h_C, h_C_ref, M, N);
  }

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_C_ref);

  return 0;
}