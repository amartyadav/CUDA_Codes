#include <cstdio>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>


#define CUDA_CHECK(call) \
    {                    \
            cudaError_t err = call; \
            if(err != cudaSuccess) { \
                std::cerr << "CUDA ERROR: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
                exit(1); \
            } \
}

__global__ void matrixMulNaive(const float *A, const float *B, float *C, int N) {
    // 1. Calculate global row and column indices
    // // Hint: blockIdx.x matches the x-axis (columns),
    // blockIdx.y matches y-axis (rows)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 2. Boundary check
    if (row < N && col < N) {
        float sum = 0.0f;

        // 3. The Dot Product Loop
        // Iterate k from 0 to N-1
        // Multiply A[row][k] * B[k][col]
        // Remember to linearize the indices! (e.g., A[row * N + k])
        for(int k = 0; k < N; ++k) {
            float a_val = A[row * N + k];
            float b_val = B[k * N + col];
            sum += a_val * b_val;
        }

        // 4. Write back to global memory
        // TODO: Store sum into C at the correct Linear index
        C[row * N + col] = sum;
    }
}


int main() {
    int N = 1024;
    size_t bytes = N * N * sizeof(float);

    // host memory allocation
    std::vector<float> h_a(N * N, 1.0f);
    std::vector<float> h_b(N * N, 2.0f);
    std::vector<float> h_c(N * N);

    // Device memory allocation
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // copying from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes
        , cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes
        , cudaMemcpyHostToDevice));

    int blockSize = 16;

    dim3 block(blockSize, blockSize);
    dim3 grid(
        (N + blockSize - 1) / blockSize,
        (N + blockSize - 1) / blockSize
    );

    printf("Launching kernel with grid(%d, %d) and block (%d, %d)\n",
        grid.x, grid.y, block.x, block.y);

    matrixMulNaive<<<grid, block>>>(d_a, d_b, d_c, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // Verification (Top-left element should be 2.0 * 1024 = 2048.0)
        printf("C[0] = %f\n", h_c[0]);
        printf(h_c[0] == 2048.0f ? "Test PASSED\n" : "Test FAILED\n");

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        return 0;

}
