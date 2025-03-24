#pragma once
template <const int BLOCK_SIZE>
__global__ void sgemm_v2(const float *A, const float *B, float *C, int M, int N, int K){
  int by = blockIdx.y;
  int bx = blockIdx.x;

  int ty = threadIdx.x / BLOCK_SIZE;
  int tx = threadIdx.x % BLOCK_SIZE;


  __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

  A = &A[by * BLOCK_SIZE * K];
  B = &B[bx * BLOCK_SIZE];
  C = &C[by * BLOCK_SIZE * N + bx * BLOCK_SIZE];

  float tmp = 0.0;

  for (int k = 0; k < K; k += BLOCK_SIZE){
    As[ty * BLOCK_SIZE + tx] = A[ty * K + tx];
    Bs[ty * BLOCK_SIZE + tx] = B[ty * N + tx];
    __syncthreads();

    A += BLOCK_SIZE;
    B += BLOCK_SIZE * N;

    for (int i = 0; i < BLOCK_SIZE; i++){
      tmp += As[ty * BLOCK_SIZE + i] * Bs[i * BLOCK_SIZE + tx];
    }
    __syncthreads();

  }
  C[ty * N + tx] = tmp;
}