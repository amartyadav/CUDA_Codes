

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>

__global__ void convolution_1d_kernel(const float *input, const float *kernel, float *output,
                                      int input_size, int kernel_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;

    if (idx >= output_size)
        return;

    float sum = 0.0f;

    for (int k = 0; k < kernel_size; k++)
    {
        float input_value = input[idx + k];
        float kernel_value = kernel[k];
        sum += input_value * kernel_value;
    }

    output[idx] = sum;
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, const float *kernel, float *output, int input_size,
                      int kernel_size)
{
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
                                                              kernel_size);
    cudaDeviceSynchronize();
}

int main()
{
    auto run_test = [](const std::vector<float> &input,
                       const std::vector<float> &kernel,
                       const std::vector<float> &expected,
                       const char *test_name)
    {
        int input_size = static_cast<int>(input.size());
        int kernel_size = static_cast<int>(kernel.size());
        int output_size = input_size - kernel_size + 1;

        std::vector<float> h_output(output_size, 0.0f);

        float *d_input = nullptr;
        float *d_kernel = nullptr;
        float *d_output = nullptr;

        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_kernel, kernel_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        cudaMemcpy(d_input, input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, kernel.data(), kernel_size * sizeof(float), cudaMemcpyHostToDevice);

        solve(d_input, d_kernel, d_output, input_size, kernel_size);

        cudaMemcpy(h_output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_kernel);
        cudaFree(d_output);

        bool passed = true;
        for (int i = 0; i < output_size; ++i)
        {
            if (std::fabs(h_output[i] - expected[i]) > 1e-5f)
            {
                passed = false;
                break;
            }
        }

        std::cout << "[" << test_name << "] " << (passed ? "PASS" : "FAIL") << "\n";
        std::cout << "Expected (first up to 8): ";
        for (int i = 0; i < output_size && i < 8; ++i)
        {
            std::cout << expected[i] << " ";
        }
        std::cout << "\nActual   (first up to 8): ";
        for (int i = 0; i < output_size && i < 8; ++i)
        {
            std::cout << h_output[i] << " ";
        }
        std::cout << "\n\n";

        return passed;
    };

    std::vector<float> input1(25000, 1.0f);
    std::vector<float> kernel1(4, 0.25f);
    std::vector<float> expected1(25000 - 4 + 1, 1.0f);

    std::vector<float> input2 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> kernel2 = {1.0f, 0.0f, -1.0f};
    std::vector<float> expected2 = {-2.0f, -2.0f};

    bool pass1 = run_test(input1, kernel1, expected1, "All-ones moving average");
    bool pass2 = run_test(input2, kernel2, expected2, "Small hand-computed case");

    std::cout << "Overall: " << ((pass1 && pass2) ? "PASS" : "FAIL") << "\n";
    return (pass1 && pass2) ? 0 : 1;
}