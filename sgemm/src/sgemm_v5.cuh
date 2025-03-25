#pragma once

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
// FLOAT4优化
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_v5(float *A, float *B, float *C, int M, int N, int K) {

}