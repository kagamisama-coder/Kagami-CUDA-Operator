#pragma once
#include<cstdio>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
// FLOAT4优化 优化点主要在于将全局内存搬运到共享内存过程中，一个线程不只搬运一个元素而是搬运四个元素，减少内存事务的数量
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_v5(float *A, float *B, float *C, int M, int N, int K) {
  int bx = blockIdx.x;
  int by = blockIdx.y;

  const int block_row_thread = BN / TN;
  const int block_col_thread = BM / TM;
  const int thread_num = block_row_thread * block_col_thread;

  // 当前线程对应的thread tile的左上角元素在block中的位置
  int tx = threadIdx.x % block_row_thread * TN;
  int ty = threadIdx.x / block_row_thread * TM;

  // 每个线程搬运四个浮点数，完全搬运到As需要所有线程搬运ldg_a_num轮
  const int ldg_a_num = BM * BK / (4 * thread_num);
  const int ldg_b_num = BK * BN / (4 * thread_num);

  // 当前线程负责As[a_tile_row][a_tile_col + i] = A[a_tile_row][a_tile_col + i]的搬运，其中i=0，1，2，3
  // 当前线程搬运的第一个内存块的初始索引
  // 一行有(BK / 4)个线程来负责搬，由此计算出threadIdx.x负责的搬运元素到底在第几行第几列，注意列数是4的倍数
  int a_tile_row = threadIdx.x / (BK / 4);
  int a_tile_col = (threadIdx.x % (BK / 4)) * 4;
  int a_tile_stride = BM / ldg_a_num; // 每轮需要搬运的行数

  int b_tile_row = threadIdx.x / (BN / 4);
  int b_tile_col = (threadIdx.x % (BN / 4)) * 4;
  int b_tile_stride = BK / ldg_b_num;

  float accum[TM][TN] = {0.};
  
  // 用于转置As矩阵
  float ldg_a_reg[4 * ldg_a_num] = {0.}; // 一个线程一轮搬运4个浮点数，需要搬运ldg_a_num轮，用寄存器来缓存

  // 计算过程中缓存共享内存
  float a_frag[TM] = {0.};
  float b_frag[TN] = {0.};

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A = &A[by * BM * K];
  B = &B[bx * BN];
  C = &C[by * BM * N + bx * BN];

#pragma unroll
  for (int k = 0; k < K; k+=BK){
    #pragma unroll
    for (int i = 0; i < BM; i+=a_tile_stride){
      int ldg_index = i / a_tile_stride * 4; // 计算的是在ldg_a_reg数组中的起始位置
      FLOAT4(ldg_a_reg[ldg_index]) = FLOAT4(A[(a_tile_row + i) * K + a_tile_col]);

      // As转置存储
      As[a_tile_col * BM + a_tile_row + i] = ldg_a_reg[ldg_index];
      As[(a_tile_col + 1) * BM + a_tile_row + i] = ldg_a_reg[ldg_index + 1];
      As[(a_tile_col + 2) * BM + a_tile_row + i] = ldg_a_reg[ldg_index + 2];
      As[(a_tile_col + 3) * BM + a_tile_row + i] = ldg_a_reg[ldg_index + 3];
    }

#pragma unroll
    for (int i = 0; i < BK; i+= b_tile_stride){
      FLOAT4(Bs[(b_tile_row + i) * BN + b_tile_col]) =
          FLOAT4(B[(b_tile_row + i) * N + b_tile_col]);
    }
    __syncthreads();

    A += BK;
    B += BK * N;

    for (int i = 0; i < BK; i++){
      for (int m = 0; m < TM; m+=4){
        // 此时As已经进行过转置，格式是(BK, BM)
        FLOAT4(a_frag[m]) = FLOAT4(As[i * BM + ty + m]);
      }
      for (int n = 0; n < TN; n+=4){
        FLOAT4(b_frag[n]) = FLOAT4(Bs[i * BN + tx + n]);
      }

      for (int m = 0; m < TM; m++){
        for (int n = 0; n < TN; n++){
          accum[m][n] += a_frag[m] * b_frag[n];
        }
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int m = 0; m < TM; m++){
    #pragma unroll
    for (int n = 0; n < TN; n+=4){
      float4 ctmp = FLOAT4(C[(ty + m) * N + tx + n]);
      ctmp.x = accum[m][n];
      ctmp.y = accum[m][n + 1];
      ctmp.z = accum[m][n + 2];
      ctmp.w = accum[m][n + 3];
      FLOAT4(C[(ty + m) * N + tx + n]) = ctmp;
    }
  }
}