#pragma once
#include<cstdio>
#include "../utils.cuh"

template<const int BM, const int BN, const int BK, int TM, int TN>
__global__ void sgemm_v4(const float *A, const float *B, float *C, int M, int N, int K){
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int block_row_thread = BN / TN;
  int block_col_thread = BM / TM;
  int thread_num = block_row_thread * block_col_thread;

  int tx = (threadIdx.x % block_row_thread) * TN;
  int ty = (threadIdx.x / block_row_thread) * TM;

  int a_tile_row = threadIdx.x / BK;
  int a_tile_col = threadIdx.x % BK;
  int a_stride_num = thread_num / BK;

  int b_tile_row = threadIdx.x / BN;
  int b_tile_col = threadIdx.x % BN;
  int b_stride_num = thread_num / BN;

  A = &A[by * BM * K];
  B = &B[bx * BN];
  C = &C[by * BM * N + bx * BN];

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  float tmp[TM][TN] = {0.0};
  for (int k = 0; k < K; k += BK) {
    #pragma unroll
    for (int i = 0; i < BM; i += a_stride_num){
      As[(a_tile_row + i) * BK + a_tile_col] =
          A[(a_tile_row + i) * K + a_tile_col];
    }
#pragma unroll
    for (int i = 0; i < BK; i += b_stride_num){
      Bs[(b_tile_row + i) * BN + b_tile_col] =
          B[(b_tile_row + i) * N + b_tile_col];
    }
    __syncthreads();

    A += BK;
    B += BK * N;

    #pragma unroll
    for (int i = 0; i < TM; i++){
      #pragma unroll
      for (int j = 0; j < TN; j++){
        #pragma unroll
        for (int l = 0; l < BK; l++){
          tmp[i][j] += As[(ty + i) * BK + l] * Bs[l * BN + tx + j];
        }
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < TM; i++){
    #pragma unroll
    for (int j = 0; j < TN; j++){
      C[(ty + i) * N + tx + j] = tmp[i][j];
    }
  }
}