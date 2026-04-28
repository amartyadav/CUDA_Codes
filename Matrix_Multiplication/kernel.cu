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
