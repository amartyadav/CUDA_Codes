// Matrix Multiplication
// Easy
// Write a program that multiplies two matrices of 32-bit floating point numbers on a GPU. Given matrix A
//  of MxN dimensions 
//  and matrix B
//  of NxK dimensions 
// , compute the product matrix 
// , which will have dimensions MxK
// . All matrices are stored in row-major format.

// Implementation Requirements
// Use only native features (external libraries are not permitted)
// The solve function signature must remain unchanged
// The final result must be stored in matrix C

// (ixC+j)
#include <cuda_runtime.h>
#include <iostream>

__global__ void matrix_multiplication_kernel(const float *A, const float *B, float *C, int M, int N, int K) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0.0f;
    if (row < M && col < K)
    {
        for (int i = 0; i < N; i++){
            value += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = value;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float *A, const float *B, float *C, int M, int N, int K)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

int main()
{
    int M = 1024, N = 1024, K = 1024;

    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * K * sizeof(float);

    float *d_A, *d_B, *d_C;
    float *h_A, *h_B, *h_C;

    h_A = (float *)malloc(sizeA);
    h_B = (float *)malloc(sizeB);
    h_C = (float *)malloc(sizeC);

    for (int i = 0; i < M * N; i++)
        h_A[i] = 1.0f;
    for (int i = 0; i < N * K; i++)
        h_B[i] = 1.0f;

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    solve(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++)
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}