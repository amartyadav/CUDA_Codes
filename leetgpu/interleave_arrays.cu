// Interleave Arrays
// Easy
// Write a GPU program that interleaves two arrays of 32-bit floating point numbers. Given two input arrays A and B, each of length N, produce an output array of length 2N where elements alternate between the two inputs: [A[0], B[0], A[1], B[1], A[2], B[2], ...]

// A: a₀,a₁,a₂,a₃
// B: b₀,b₁,b₂,b₃
// output: a₀,b₀,a₁,b₁,a₂,b₂,a₃,b₃

// Implementation Requirements
// Use only native features (external libraries are not permitted)
// The solve function signature must remain unchanged
// The final result must be stored in the output array
// Example 1:
// Input:  A = [1.0, 2.0, 3.0], B = [4.0, 5.0, 6.0]
// Output: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
// Example 2:
// Input:  A = [10.0, 20.0], B = [30.0, 40.0]
// Output: [10.0, 30.0, 20.0, 40.0]

#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__global__ void interleave_kernel(const float *A, const float *B, float *output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N*2 && idx % 2 == 0)
    {
        output[idx] = A[idx/2];
    }
    if (idx < N*2 && idx % 2 != 0)
    {
        output[idx] = B[idx/2];
    }
}

// A, B, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *output, int N)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (2 * N + threadsPerBlock - 1) / threadsPerBlock;

    interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output, N);
    cudaDeviceSynchronize();
}

int main()
{
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_output = nullptr;

    int N = 5;

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_output, N * 2 * sizeof(float));

    std::vector<float> h_A = {1,3,5,7,9};
    std::vector<float> h_B = {2, 4, 6, 8, 10};
    std::vector<float> h_output(2 * N, 0.0f);

    cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_A, d_B, d_output, N);

    cudaGetLastError();

    cudaMemcpy(h_output.data(), d_output, 2 * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_output);

    for (auto i = 0; i < 2 * N; i++)
    {
        std::cout << h_output[i] << ", ";
    }
    std::cout << "\n";
}