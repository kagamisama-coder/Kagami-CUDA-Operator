#pragma once
#include <cstdio>

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_v3(const float *A, const float *B, float *C, int M, int N,
                         int K) {
  int by = blockIdx.y;
  int bx = blockIdx.x;

  int ty = threadIdx.x / BN * TM;
  int tx = threadIdx.x % BN;

  int block_row_thread = BN;
  int block_col_thread = BM / TM;
  int thread_num = block_row_thread * block_col_thread;

  int a_tile_row = threadIdx.x / BK;
  int a_tile_col = threadIdx.x % BK;
  int a_stride_num = thread_num / BK;

  int b_tile_row = threadIdx.x / BN;
  int b_tile_col = threadIdx.x % BN;
  int b_stride_num = thread_num / BN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A = &A[by * BM * K];
  B = &B[bx * BN];
  C = &C[by * BM * N + bx * BN];

  float tmp[TM + 1] = {0.0};
#pragma unroll
  for (int k = 0; k < K; k += BK) {
#pragma unroll
    for (int i = 0; i < BM; i += a_stride_num) {
      As[(i + a_tile_row) * BK + a_tile_col] =
          A[(i + a_tile_row) * K + a_tile_col];
    }

#pragma unroll
    for (int i = 0; i < BK; i += b_stride_num) {
      Bs[(i + b_tile_row) * BN + b_tile_col] =
          B[(i + b_tile_row) * N + b_tile_col];
    }
    __syncthreads();
    A += BK;
    B += BK * N;

#pragma unroll
    for (int i = 0; i < TM; i++) {
      for (int j = 0; j < BK; j++) {
        // 这个代码按照矩阵乘法的逐个逻辑来写，比较通顺。如果交换循环变量的次序i和j，则可以在二层循环前取出Bs中的值，进一步降低读取共享内存的次数
        tmp[i] += As[(ty + i) * BK + j] * Bs[j * BN + tx];
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; i++) {
    C[(ty + i) * N + tx] = tmp[i];
  }
}