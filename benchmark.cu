// Reuse existing kernels; rename their standalone entry points.
#define main naive_demo_main
#include "naive_gemm.cu"
#undef main

#define main tiled_demo_main
#include "tiled_gemm.cu"
#undef main

#include <chrono>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <string>

__declspec(noinline)
void cpu_gemm(const float* a, const float* b, float* c,
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

struct Stats {
    double mean;
    double median;
};

Stats summarize(std::vector<double> values) {
    double mean =
        std::accumulate(values.begin(), values.end(), 0.0)
        / values.size();
    std::sort(values.begin(), values.end());
    size_t mid = values.size() / 2;
    double median = values.size() % 2
        ? values[mid] : (values[mid - 1] + values[mid]) / 2.0;
    return {mean, median};
}

struct Errors {
    double max_abs = 0.0;
    double mae = 0.0;
    double rmse = 0.0;
    int mismatches = 0;
};

Errors compare(const std::vector<float>& result,
               const std::vector<float>& reference) {
    Errors e;
    double squared = 0.0;

    for (size_t i = 0; i < result.size(); ++i) {
        if (!std::isfinite(result[i]))
            throw std::runtime_error("Non-finite GPU output");

        double delta =
            std::abs(double(result[i]) - double(reference[i]));

        e.max_abs = std::max(e.max_abs, delta);
        e.mae += delta;
        squared += delta * delta;

        if (delta > 1e-5 + 1e-5 * std::abs(double(reference[i])))
            ++e.mismatches;
    }

    e.mae /= result.size();
    e.rmse = std::sqrt(squared / result.size());
    return e;
}

void launch(int tile, const float* a, const float* b, float* c,
            int m, int n, int k) {
    int side = tile == 0 ? 16 : tile;
    dim3 block(side, side);
    dim3 grid((n + side - 1) / side, (m + side - 1) / side);

    switch (tile) {
        case 0:
            naive_gemm<<<grid, block>>>(a, b, c, m, n, k);
            break;
        case 8:
            tiled_gemm<8><<<grid, block>>>(a, b, c, m, n, k);
            break;
        case 16:
            tiled_gemm<16><<<grid, block>>>(a, b, c, m, n, k);
            break;
        case 32:
            tiled_gemm<32><<<grid, block>>>(a, b, c, m, n, k);
            break;
    }
}

void write_row(std::ofstream& csv, const char* scene,
               const std::string& name, int m, int n, int k,
               Stats stats, double cpu_ms, double naive_ms,
               Errors error) {
    double gflops = 2.0 * m * n * k / (stats.mean * 1e6);

    csv << scene << ',' << name << ",fp32,"
        << m << ',' << n << ',' << k << ','
        << stats.mean << ',' << stats.median << ','
        << gflops << ',' << cpu_ms / stats.mean << ',';

    if (naive_ms > 0.0)
        csv << naive_ms / stats.mean;

    csv << ',' << error.max_abs << ',' << error.mae << ','
        << error.rmse << ',' << error.mismatches << '\n';

    std::printf(
        "%-12s mean=%9.6f ms median=%9.6f ms "
        "GFLOPS=%8.3f CPU_speedup=%7.3fx errors=%d\n",
        name.c_str(), stats.mean, stats.median,
        gflops, cpu_ms / stats.mean, error.mismatches);
}

bool run_case(const char* scene, int m, int n, int k,
              std::ofstream& csv, std::ofstream& samples) {
    constexpr int repeats = 100;
    constexpr int warmup = 10;

    std::vector<float> a(size_t(m) * k);
    std::vector<float> b(size_t(k) * n);
    std::vector<float> reference(size_t(m) * n);
    std::vector<float> output(size_t(m) * n);

    if (n == 1 && k == 9 && m == 676) {
        // Exactly the original input and Im2col mapping.
        float input[28 * 28];
        for (int i = 0; i < 28 * 28; ++i)
            input[i] = float((i * 17 + 3) % 101 - 50) / 50.0f;

        for (int y = 0; y < 26; ++y)
            for (int x = 0; x < 26; ++x)
                for (int ky = 0; ky < 3; ++ky)
                    for (int kx = 0; kx < 3; ++kx)
                        a[(y * 26 + x) * 9 + ky * 3 + kx] =
                            input[(y + ky) * 28 + x + kx];

        for (int i = 0; i < 9; ++i)
            b[i] = float((i * 7 + 1) % 17 - 8) / 8.0f;
    } else {
        // Deterministic synthetic matrices with convolution-derived shapes.
        unsigned state = 12345;
        auto next_value = [&]() {
            state = 1664525u * state + 1013904223u;
            return float(int((state >> 16) % 2001u) - 1000)
                   / 1000.0f;
        };
        for (float& value : a) value = next_value();
        for (float& value : b) value = next_value();
    }

    cpu_gemm(a.data(), b.data(), reference.data(), m, n, k);

    volatile float sink = 0.0f;
    for (int i = 0; i < warmup; ++i) {
        cpu_gemm(a.data(), b.data(), output.data(), m, n, k);
        sink = output[i % output.size()];
    }

    // Amortize host timer overhead for the tiny original workload.
    int inner = n == 1 ? 1000 : 1;
    std::vector<double> cpu_times;

    for (int r = 0; r < repeats; ++r) {
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < inner; ++i) {
            cpu_gemm(a.data(), b.data(), output.data(), m, n, k);
            sink = output[i % output.size()];
        }
        auto end = std::chrono::steady_clock::now();

        double ms =
            std::chrono::duration<double, std::milli>(end - start)
            .count() / inner;

        cpu_times.push_back(ms);
        samples << scene << ",cpu," << r << ',' << inner
                << ',' << ms << '\n';
    }

    Stats cpu_stats = summarize(cpu_times);
    std::printf("\n%s: M=%d N=%d K=%d\n", scene, m, n, k);
    write_row(csv, scene, "cpu", m, n, k,
              cpu_stats, cpu_stats.mean, 0.0, {});

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&da),
                     a.size() * sizeof(float)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&db),
                     b.size() * sizeof(float)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&dc),
                     output.size() * sizeof(float)));

    CHECK(cudaMemcpy(da, a.data(), a.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, b.data(), b.size() * sizeof(float),
                     cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    double naive_ms = 0.0;
    bool passed = true;

    for (int tile : {0, 8, 16, 32}) {
        std::string name =
            tile == 0 ? "naive" : "tiled_" + std::to_string(tile);

        CHECK(cudaMemset(dc, 0xff, output.size() * sizeof(float)));
        launch(tile, da, db, dc, m, n, k);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(output.data(), dc,
                         output.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));

        Errors error = compare(output, reference);
        if (error.mismatches != 0) {
            std::printf("%s FAILED correctness: %d mismatches\n",
                        name.c_str(), error.mismatches);
            passed = false;
            continue;
        }

        for (int i = 0; i < warmup; ++i)
            launch(tile, da, db, dc, m, n, k);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        std::vector<double> gpu_times;
        for (int r = 0; r < repeats; ++r) {
            CHECK(cudaEventRecord(start));
            launch(tile, da, db, dc, m, n, k);
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            CHECK(cudaGetLastError());

            float ms = 0.0f;
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            gpu_times.push_back(ms);

            samples << scene << ',' << name << ',' << r
                    << ",1," << ms << '\n';
        }

        Stats stats = summarize(gpu_times);
        if (tile == 0) naive_ms = stats.mean;

        write_row(csv, scene, name, m, n, k,
                  stats, cpu_stats.mean, naive_ms, error);
    }

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(da));
    CHECK(cudaFree(db));
    CHECK(cudaFree(dc));
    return passed;
}

int main() {
    try {
        CHECK(cudaSetDevice(0));
        cudaDeviceProp prop{};
        CHECK(cudaGetDeviceProperties(&prop, 0));

        std::printf("GPU: %s\n", prop.name);
        std::puts("10 warmups, 100 samples per implementation.");
        std::puts("GEMM only; allocation, Im2col and transfers excluded.");

        std::ofstream csv("results/cuda_benchmark.csv");
        std::ofstream samples("results/benchmark_samples.csv");
        if (!csv || !samples)
            throw std::runtime_error("Create the results directory first");

        csv << std::setprecision(12);
        samples << std::setprecision(12);

        csv << "scene,implementation,precision,M,N,K,mean_ms,"
               "median_ms,gflops,speedup_vs_cpu,speedup_vs_naive,"
               "max_abs_error,mae,rmse,mismatches\n";

        samples << "scene,implementation,sample,inner_iterations,"
                   "per_gemm_ms\n";

        bool original = run_case("original", 676, 1, 9, csv, samples);
        bool expanded = run_case("expanded", 676, 64, 144, csv, samples);

        std::puts(original && expanded
            ? "\nALL CORRECTNESS CHECKS PASSED"
            : "\nFAILED: do not use results as a passing benchmark");

        return original && expanded ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}