// Implement the SiLU (Sigmoid Linear Unit) activation function forward pass for 1D input vectors. Given an input tensor of shape [N] where N is the number of elements, compute the output using the elementwise formula.

// SiLU is defined as:

// \begin { align }
// \sigma(x) &= \frac{1}
// {
//     1 + e ^ { -x }
// }
// \\
//   \text{SiLU}(x) &= x \cdot \sigma(x)
//   \end { align }

// Implementation Requirements
// Use only native features (external libraries are not permitted)
// The solve function signature must remain unchanged
// The final result must be stored in the output tensor
// Example 1:
// Input:  input = [0.5, 1.0, -0.5]  (N=3)
// Output: output = [0.3112295, 0.731059, -0.1887705]
// Example 2:
// Input:  input = [-1.0, -2.0, -3.0, -4.0, -5.0]  (N=5)
// Output: output = [-0.26894143 -0.23840584 -0.14227763 -0.07194484 -0.03346425]
// Constraints
// 1 ≤ N ≤ 10,000
// -100.0 ≤ input values ≤ 100.0
// Performance is measured with N = 50,000

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>

__global__ void silu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N)
    {
        float sigma_x = 1 / (1 + std::exp(input[idx] * - 1));
        output[idx] = input[idx] * sigma_x;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

// ====================== CPU REFERENCE ======================
void cpu_silu(const float *input, float *output, int N)
{
    for (int i = 0; i < N; i++)
    {
        float x = input[i];
        float sigma = 1.0f / (1.0f + expf(-x));
        output[i] = x * sigma;
    }
}

// ====================== TEST HELPER ======================
bool run_test(const std::string &name, const std::vector<float> &h_input, bool check_perf = false)
{
    int N = h_input.size();
    std::vector<float> h_output(N);
    std::vector<float> h_ref(N);

    float *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Run GPU kernel
    solve(d_input, d_output, N);

    cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute reference
    cpu_silu(h_input.data(), h_ref.data(), N);

    // Compare
    bool passed = true;
    const float atol = 1e-5f;
    const float rtol = 1e-5f;

    std::cout << "\n=== Test: " << name << " (N=" << N << ") ===\n";
    for (int i = 0; i < N && i < 10; i++)
    { // show first 10 elements
        float diff = fabs(h_output[i] - h_ref[i]);
        bool ok = diff <= atol || diff <= rtol * fabs(h_ref[i]);
        if (!ok)
            passed = false;

        if (i < 5 || !ok)
        {
            std::cout << "  [" << i << "] GPU: " << std::fixed << std::setprecision(7)
                      << h_output[i] << " | Ref: " << h_ref[i]
                      << (ok ? "  ✓" : "  ✗") << "\n";
        }
    }

    if (passed)
        std::cout << "✅ PASSED\n";
    else
        std::cout << "❌ FAILED\n";

    cudaFree(d_input);
    cudaFree(d_output);
    return passed;
}

// ====================== MAIN TEST SUITE ======================
int main()
{
    std::cout << "LeetGPU - SiLU Local Test Suite\n";
    std::cout << "===============================\n";

    bool all_passed = true;

    // 1. Example from problem statement
    all_passed &= run_test("Example 1", {0.5f, 1.0f, -0.5f});

    // 2. Single element
    all_passed &= run_test("Single Element", {2.0f});

    // 3. All zeros
    {
        std::vector<float> zeros(42, 0.0f);
        all_passed &= run_test("All Zeros", zeros);
    }

    // 4. Negative numbers
    all_passed &= run_test("Negative Numbers", {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f});

    // 5. Mixed positive/negative
    all_passed &= run_test("Mixed Pos/Neg", {-0.5f, 0.0f, 0.5f, 1.0f});

    // 6. Large values (-100 to 100)
    {
        std::vector<float> large(1024);
        for (int i = 0; i < 1024; i++)
        {
            large[i] = -100.0f + (200.0f * i) / 1023.0f; // evenly spaced
        }
        all_passed &= run_test("Large Values (-100 to 100)", large);
    }

    // 7. Large N (functional)
    {
        std::vector<float> largeN(10000);
        for (int i = 0; i < 10000; i++)
        {
            largeN[i] = -50.0f + (100.0f * i) / 9999.0f;
        }
        all_passed &= run_test("Large N = 10000", largeN);
    }

    // 8. Performance Test (N=50000)
    {
        std::vector<float> perf(50000);
        for (int i = 0; i < 50000; i++)
        {
            perf[i] = -50.0f + (100.0f * (i % 997)) / 996.0f; // pseudo-random
        }
        std::cout << "\n=== Performance Test (N=50000) ===\n";
        run_test("Performance Test N=50000", perf, true);
    }

    std::cout << "\n===============================\n";
    if (all_passed)
        std::cout << "🎉 ALL TESTS PASSED!\n";
    else
        std::cout << "❌ SOME TESTS FAILED\n";

    return 0;
}