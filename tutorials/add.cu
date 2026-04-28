#include <iostream>
#include "math.h"

// kernel function to add two arrays' elements
__global__
void add(int N, float* X, float *Y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i+=stride)
    {
        Y[i] = X[i] + Y[i];
    }
}

int main(void)
{
    int N = 1 << 20;
    float *x, *y, *sum;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));


    // initialise x and y on host
    for (int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaMemLocation location;
    location.type = cudaMemLocationTypeDevice;
    location.id = 0;
    cudaMemPrefetchAsync(x, N * sizeof(float), location, 0);
    cudaMemPrefetchAsync(y, N * sizeof(float), location, 0);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // running the kernel on CPU
    add<<<numBlocks, blockSize>>>(N, x, y);

    cudaDeviceSynchronize();

    // checking for error
    float maxError = 0.0f;
    for (int i = 0; i < N; i++){
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }

    std::cout << "Max Error: " << maxError << std::endl;

    // free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}