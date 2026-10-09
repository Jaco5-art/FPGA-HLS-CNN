#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#define CHECK(call) do {                                   \
    cudaError_t err = (call);                              \
    if (err != cudaSuccess) {                              \
        std::fprintf(stderr, "%s: %s\n", #call,             \
                     cudaGetErrorString(err));              \
        std::exit(1);                                      \
    }                                                     \
} while (0)

// 一个线程计算 C 的一个元素；只使用 global memory。
__global__ void naive_gemm(const float* a, const float* b,
                           float* c, int m, int n, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int p = 0; p < k; ++p)
            sum += a[row * k + p] * b[p * n + col];
        c[row * n + col] = sum;
    }
}

int main() {
    constexpr int H = 28, W = 28;
    constexpr int OH = 26, OW = 26;
    constexpr int M = OH * OW, N = 1, K = 9;

    std::vector<float> input(H * W), b(K * N), a(M * K);
    std::vector<float> cpu(M * N), gpu(M * N);

    for (int i = 0; i < H * W; ++i)
        input[i] = float((i * 17 + 3) % 101 - 50) / 50.0f;
    for (int i = 0; i < K; ++i)
        b[i] = float((i * 7 + 1) % 17 - 8) / 8.0f;

    // CPU Im2col：与上一阶段一致。
    for (int y = 0; y < OH; ++y)
        for (int x = 0; x < OW; ++x)
            for (int ky = 0; ky < 3; ++ky)
                for (int kx = 0; kx < 3; ++kx)
                    a[(y * OW + x) * K + ky * 3 + kx] =
                        input[(y + ky) * W + x + kx];

    // CPU FP32 reference。
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p)
                sum += a[row * K + p] * b[p * N + col];
            cpu[row * N + col] = sum;
        }
    }

    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    const size_t bytes_a = a.size() * sizeof(float);
    const size_t bytes_b = b.size() * sizeof(float);
    const size_t bytes_c = gpu.size() * sizeof(float);

    CHECK(cudaMalloc(reinterpret_cast<void**>(&da), bytes_a));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&db), bytes_b));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&dc), bytes_c));

    CHECK(cudaMemcpy(da, a.data(), bytes_a, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, b.data(), bytes_b, cudaMemcpyHostToDevice));

    // 初始化为 NaN，漏写的输出也会被正确性检查发现。
    CHECK(cudaMemset(dc, 0xff, bytes_c));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    naive_gemm<<<grid, block>>>(da, db, dc, M, N, K);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(gpu.data(), dc, bytes_c, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(da));
    CHECK(cudaFree(db));
    CHECK(cudaFree(dc));

    double max_error = 0.0, mae = 0.0, squared_error = 0.0;
    int mismatches = 0;

    for (int i = 0; i < M * N; ++i) {
        if (!std::isfinite(gpu[i])) {
            std::fprintf(stderr, "FAIL: non-finite output at %d\n", i);
            return 1;
        }

        double error = std::abs(double(gpu[i]) - double(cpu[i]));
        max_error = std::max(max_error, error);
        mae += error;
        squared_error += error * error;

        if (error > 1e-5 + 1e-5 * std::abs(double(cpu[i])))
            ++mismatches;
    }

    mae /= M * N;
    double rmse = std::sqrt(squared_error / (M * N));

    std::ofstream report("results/naive_correctness.json");
    if (!report) {
        std::fprintf(stderr, "Cannot write results/naive_correctness.json\n");
        return 1;
    }

    report.precision(12);
    report << "{\n"
           << "  \"implementation\": \"naive_cuda_fp32\",\n"
           << "  \"reference\": \"cpu_gemm_fp32\",\n"
           << "  \"M\": " << M << ", \"N\": " << N
           << ", \"K\": " << K << ",\n"
           << "  \"outputs\": " << M * N << ",\n"
           << "  \"max_abs_error\": " << max_error << ",\n"
           << "  \"mae\": " << mae << ",\n"
           << "  \"rmse\": " << rmse << ",\n"
           << "  \"mismatches\": " << mismatches << ",\n"
           << "  \"passed\": "
           << (mismatches == 0 ? "true" : "false") << "\n}\n";

    std::printf("GPU: %s\n", prop.name);
    std::printf("GEMM: [%d x %d] * [%d x %d]\n", M, K, K, N);
    std::printf("Block: (%u, %u), Grid: (%u, %u)\n",
                block.x, block.y, grid.x, grid.y);
    std::printf("Correctness: %s | mismatches: %d/%d\n",
                mismatches == 0 ? "PASS" : "FAIL", mismatches, M * N);
    std::printf("Max error: %.9g | MAE: %.9g | RMSE: %.9g\n",
                max_error, mae, rmse);
    std::puts("Saved results/naive_correctness.json");

    return mismatches == 0 ? 0 : 1;
}