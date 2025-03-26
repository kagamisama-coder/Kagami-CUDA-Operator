#pragma once

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

// 双缓冲预取优化，最终优化版本
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_v6(float *A, float *B, float *C, int M, int N, int K){

}