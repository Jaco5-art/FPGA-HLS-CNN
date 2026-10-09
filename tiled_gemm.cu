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

// Shared-memory tiled GEMM: C[M,N] = A[M,K] * B[K,N].
template<int TILE>
__global__ void tiled_gemm(const float* a, const float* b,
                           float* c, int m, int n, int k) {
    __shared__ float tile_a[TILE][TILE];
    __shared__ float tile_b[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int base = 0; base < k; base += TILE) {
        int a_col = base + tx;
        int b_row = base + ty;

        // Load valid elements; pad out-of-bounds elements with zero.
        tile_a[ty][tx] = (row < m && a_col < k)
            ? a[row * k + a_col] : 0.0f;

        tile_b[ty][tx] = (b_row < k && col < n)
            ? b[b_row * n + col] : 0.0f;

        // All threads must finish loading before computation.
        __syncthreads();

        for (int p = 0; p < TILE; ++p)
            sum += tile_a[ty][p] * tile_b[p][tx];

        // Finish reading this tile before loading the next one.
        __syncthreads();
    }

    if (row < m && col < n)
        c[row * n + col] = sum;
}

int main() {
    constexpr int H = 28, W = 28;
    constexpr int OH = 26, OW = 26;
    constexpr int M = OH * OW, N = 1, K = 9;
    constexpr int TILE = 16;

    std::vector<float> input(H * W);
    std::vector<float> b(K * N), a(M * K);
    std::vector<float> cpu(M * N), gpu(M * N);

    // Same deterministic input and weights as the CPU/naive versions.
    for (int i = 0; i < H * W; ++i)
        input[i] = float((i * 17 + 3) % 101 - 50) / 50.0f;

    for (int i = 0; i < K; ++i)
        b[i] = float((i * 7 + 1) % 17 - 8) / 8.0f;

    // CPU Im2col: one flattened 3x3 window per row.
    for (int y = 0; y < OH; ++y)
        for (int x = 0; x < OW; ++x)
            for (int ky = 0; ky < 3; ++ky)
                for (int kx = 0; kx < 3; ++kx)
                    a[(y * OW + x) * K + ky * 3 + kx] =
                        input[(y + ky) * W + x + kx];

    // CPU FP32 GEMM reference.
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

    float* da = nullptr;
    float* db = nullptr;
    float* dc = nullptr;

    const size_t bytes_a = a.size() * sizeof(float);
    const size_t bytes_b = b.size() * sizeof(float);
    const size_t bytes_c = gpu.size() * sizeof(float);

    CHECK(cudaMalloc(reinterpret_cast<void**>(&da), bytes_a));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&db), bytes_b));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&dc), bytes_c));

    CHECK(cudaMemcpy(
        da, a.data(), bytes_a, cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(
        db, b.data(), bytes_b, cudaMemcpyHostToDevice));

    // Initialize output to NaN to detect missing writes.
    CHECK(cudaMemset(dc, 0xff, bytes_c));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE,
              (M + TILE - 1) / TILE);

    tiled_gemm<TILE><<<grid, block>>>(da, db, dc, M, N, K);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(
        gpu.data(), dc, bytes_c, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(da));
    CHECK(cudaFree(db));
    CHECK(cudaFree(dc));

    double max_error = 0.0;
    double mae = 0.0;
    double squared_error = 0.0;
    int mismatches = 0;

    for (int i = 0; i < M * N; ++i) {
        if (!std::isfinite(gpu[i])) {
            std::fprintf(
                stderr, "FAIL: non-finite output at %d\n", i);
            return 1;
        }

        double error =
            std::abs(double(gpu[i]) - double(cpu[i]));

        max_error = std::max(max_error, error);
        mae += error;
        squared_error += error * error;

        double tolerance =
            1e-5 + 1e-5 * std::abs(double(cpu[i]));

        if (error > tolerance)
            ++mismatches;
    }

    mae /= M * N;
    double rmse = std::sqrt(squared_error / (M * N));

    std::ofstream report("results/tiled_correctness.json");
    if (!report) {
        std::fprintf(
            stderr, "Cannot write results/tiled_correctness.json\n");
        return 1;
    }

    report.precision(12);
    report << "{\n"
           << "  \"implementation\": \"tiled_cuda_fp32\",\n"
           << "  \"reference\": \"cpu_gemm_fp32\",\n"
           << "  \"tile_size\": " << TILE << ",\n"
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
    std::printf(
        "GEMM: [%d x %d] * [%d x %d]\n", M, K, K, N);
    std::printf("Tile size: %d x %d\n", TILE, TILE);
    std::printf(
        "Block: (%u, %u), Grid: (%u, %u)\n",
        block.x, block.y, grid.x, grid.y);
    std::printf(
        "Correctness: %s | mismatches: %d/%d\n",
        mismatches == 0 ? "PASS" : "FAIL",
        mismatches, M * N);
    std::printf(
        "Max error: %.9g | MAE: %.9g | RMSE: %.9g\n",
        max_error, mae, rmse);
    std::puts("Saved results/tiled_correctness.json");

    return mismatches == 0 ? 0 : 1;
}