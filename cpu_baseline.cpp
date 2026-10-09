#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

constexpr int H = 28, W = 28;
constexpr int KH = 3, KW = 3;
constexpr int OH = H - KH + 1, OW = W - KW + 1;
constexpr int M = OH * OW, K = KH * KW, N = 1;

// 每行保存一个 3x3 输入窗口。
void im2col(const float* input, float* a) {
    for (int y = 0; y < OH; ++y)
        for (int x = 0; x < OW; ++x)
            for (int ky = 0; ky < KH; ++ky)
                for (int kx = 0; kx < KW; ++kx)
                    a[(y * OW + x) * K + ky * KW + kx] =
                        input[(y + ky) * W + x + kx];
}

// 普通单线程 FP32 GEMM，不调用 BLAS。
// noinline 配合后面的结果读取，避免重复计算被优化掉。
__declspec(noinline)
void gemm(const float* a, const float* b, float* c,
          int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p)
                sum += a[i * k + p] * b[p * n + j];
            c[i * n + j] = sum;
        }
    }
}

// 直接计算卷积，使用 double 累加校验 FP32 GEMM。
void direct_conv(const float* input, const float* weight,
                 double* output) {
    for (int y = 0; y < OH; ++y) {
        for (int x = 0; x < OW; ++x) {
            double sum = 0.0;
            for (int ky = 0; ky < KH; ++ky)
                for (int kx = 0; kx < KW; ++kx)
                    sum += double(input[(y + ky) * W + x + kx])
                         * double(weight[ky * KW + kx]);
            output[y * OW + x] = sum;
        }
    }
}

int main() {
    std::vector<float> input(H * W), weight(K);
    std::vector<float> a(M * K), output(M * N);
    std::vector<double> reference(M);

    // 固定、可复现的数据；不依赖原模型权重。
    for (int i = 0; i < H * W; ++i)
        input[i] = float((i * 17 + 3) % 101 - 50) / 50.0f;
    for (int i = 0; i < K; ++i)
        weight[i] = float((i * 7 + 1) % 17 - 8) / 8.0f;

    im2col(input.data(), a.data());
    direct_conv(input.data(), weight.data(), reference.data());
    gemm(a.data(), weight.data(), output.data(), M, N, K);

    double max_error = 0.0, mae = 0.0, squared_error = 0.0;
    bool passed = true;
    for (int i = 0; i < M; ++i) {
        double error = std::abs(double(output[i]) - reference[i]);
        max_error = std::max(max_error, error);
        mae += error;
        squared_error += error * error;

        if (!std::isfinite(output[i]) ||
            error > 1e-5 + 1e-5 * std::abs(reference[i]))
            passed = false;
    }
    mae /= M;
    double rmse = std::sqrt(squared_error / M);

    constexpr int warmup = 20;
    constexpr int repeats = 100;
    constexpr int inner = 1000;
    volatile float sink = 0.0f;

    for (int i = 0; i < warmup; ++i) {
        gemm(a.data(), weight.data(), output.data(), M, N, K);
        sink = output[i % M];
    }

    // 原始矩阵很小：每组运行 1000 次，再折算每次耗时。
    // 共记录 100 组，计时范围只包含 GEMM 和轻量结果读取。
    std::vector<double> times;
    for (int r = 0; r < repeats; ++r) {
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < inner; ++i) {
            gemm(a.data(), weight.data(), output.data(), M, N, K);
            sink = output[i % M];
        }
        auto end = std::chrono::steady_clock::now();
        double ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms / inner);
    }

    double mean = 0.0;
    for (double ms : times) mean += ms;
    mean /= repeats;

    auto sorted = times;
    std::sort(sorted.begin(), sorted.end());
    double median = (sorted[49] + sorted[50]) / 2.0;
    double flops = 2.0 * M * N * K;
    double gflops = mean > 0.0 ? flops / (mean * 1e6) : 0.0;

    std::ofstream csv("results/cpu_baseline.csv");
    std::ofstream raw("results/cpu_samples.csv");
    std::ofstream check("results/cpu_correctness.json");
    if (!csv || !raw || !check) {
        std::fprintf(stderr, "Cannot write results. Create results directory.\n");
        return 1;
    }

    csv.precision(12);
    csv << "implementation,precision,M,N,K,warmup,repeats,"
           "inner_iterations,mean_ms,median_ms,gflops,correct\n";
    csv << "cpu_naive,fp32," << M << ',' << N << ',' << K << ','
        << warmup << ',' << repeats << ',' << inner << ','
        << mean << ',' << median << ',' << gflops << ','
        << (passed ? "true" : "false") << '\n';

    raw.precision(12);
    raw << "sample,per_gemm_ms\n";
    for (int i = 0; i < repeats; ++i)
        raw << i << ',' << times[i] << '\n';

    check.precision(12);
    check << "{\n"
          << "  \"reference\": \"direct_conv_fp64_accumulation\",\n"
          << "  \"outputs\": " << M << ",\n"
          << "  \"max_abs_error\": " << max_error << ",\n"
          << "  \"mae\": " << mae << ",\n"
          << "  \"rmse\": " << rmse << ",\n"
          << "  \"passed\": " << (passed ? "true" : "false") << "\n}\n";

    std::printf("GEMM: [%d x %d] * [%d x %d]\n", M, K, K, N);
    std::printf("Correctness: %s (%d outputs)\n",
                passed ? "PASS" : "FAIL", M);
    std::printf("Max error: %.9g | MAE: %.9g | RMSE: %.9g\n",
                max_error, mae, rmse);
    std::printf("Mean: %.6f ms | Median: %.6f ms | %.4f GFLOPS\n",
                mean, median, gflops);
    std::puts("Saved results/cpu_baseline.csv, cpu_samples.csv, cpu_correctness.json");
    return passed ? 0 : 1;
}