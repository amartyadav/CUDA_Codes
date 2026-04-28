// Matrix Addition
// Easy
// Implement a program that performs element-wise addition of two 
//  matrices containing 32-bit floating point numbers on a GPU. The program should take two input matrices of equal dimensions and produce a single output matrix containing their element-wise sum.

// Implementation Requirements
// External libraries are not permitted
// The solve function signature must remain unchanged
// The final result must be stored in matrix C

#include <cuda_runtime.h>

__global__ void matrix_add(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N*N) {
        C[idx] = A[idx] + B[idx];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int N)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
