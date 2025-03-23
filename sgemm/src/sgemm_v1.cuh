#pragma once
__global__ void sgemm_v1(const float *A, const float *B, float *C, int M, int N, int K){
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  if(ty < M && tx < N){
    float tmp = 0;
    for (int i = 0; i < K; i++) {
      tmp += A[ty * K + i] * B[i * N + tx];
    }
    C[ty * N + tx] = tmp;
  }
}
