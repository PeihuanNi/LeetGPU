#include "solve.h"
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_y = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx_x < K & idx_y < M) {
        for (int i = 0; i < N; i++) {
            C[idx_y * K + idx_x] += A[idx_y * N + i] * B[i * K + idx_x];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
