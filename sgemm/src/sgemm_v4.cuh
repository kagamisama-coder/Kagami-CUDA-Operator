#pragma once

template<const int BM, const int BN, const int BK, int TM, int TN>
__global__ void sgemm_v4(const float *A, const float *B, float *C, int M, int N, int K){
  int bx = blockIdx.x;
  int by = blockIdx.y;

  
}