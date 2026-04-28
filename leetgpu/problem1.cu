// Vector Addition
// Easy
// Write a GPU program that performs element-wise addition of two vectors containing 32-bit floating point numbers. The program should take two input vectors of equal length and produce a single output vector containing their sum.

// Implementation Requirements
// External libraries are not permitted
// The solve function signature must remain unchanged
// The final result must be stored in vector C
// Example 1:
// Input:  A = [1.0, 2.0, 3.0, 4.0]
//         B = [5.0, 6.0, 7.0, 8.0]
// Output: C = [6.0, 8.0, 10.0, 12.0]
// Example 2:
// Input:  A = [1.5, 1.5, 1.5]
//         B = [2.3, 2.3, 2.3]
// Output: C = [3.8, 3.8, 3.8]
// Constraints
// Input vectors A and B have identical lengths
// 1 ≤ N ≤ 100,000,000
// Performance is measured with N = 25,000,000

#include <iostream>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride){
        C[i] = A[i] + B[i];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

// call
int main() {
    int N = 1 << 20;
    float *d_A, *d_B, *d_C;  // device pointers
    float *h_A, *h_B, *h_C;  // host pointers
    
    // Allocate device memory
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    
    // Allocate host memory
    h_A = (float*)malloc(N * sizeof(float));
    h_B = (float*)malloc(N * sizeof(float));
    h_C = (float*)malloc(N * sizeof(float));

    // init A and B on host
    for(int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Copy host data to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_A, d_B, d_C, N);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    // h_C now contains the result of A + B

    // print first 10 results
    for(int i = 0; i < 10; i++) {
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
    }

    // free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}