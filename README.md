# CNN Convolution Acceleration with FPGA/HLS and CUDA

**基于 FPGA/HLS 与 CUDA 的 CNN 卷积加速与性能分析**

This project studies convolution acceleration through a common **Im2col + GEMM** formulation. The FPGA/HLS path explores pipelining and INT8 arithmetic; the CUDA path compares global-memory and shared-memory GEMM kernels, tile sizes, and measured GPU resource usage. The focus is operator-level correctness and architecture-aware performance analysis.

## Workloads

| Case | GEMM A × B | Purpose |
|---|---|---|
| Original | [676 × 9] × [9 × 1] | Valid 3×3 convolution on a 28×28 single-channel input |
| Expanded | [676 × 144] × [144 × 64] | Synthetic matrices with dimensions corresponding to 16 input channels and 64 output channels |

Current operator tests exclude bias, activation, pooling and dense layers. The expanded case is a convolution-derived GEMM workload, not a trained CNN evaluation. CUDA inputs and HLS quantization vectors are verified against their respective references; these are not identical cross-platform input datasets.

## FPGA/HLS path

- `Im2col()` maps image patches to a 676×9 matrix; `GemmConv2d0()` computes the convolution using GEMM.
- FP32 and INT8 implementations use HLS pipeline directives. Requested loop II is not a guarantee of achieved II or top-level latency.
- Signed INT8 inputs/weights, INT32 accumulation and output, nearest-even quantization with clipping, and explicit dequantization.
- Input and weight scales: 1/127; output dequantization scale: 1/16129. Output requantization to INT8 is not implemented.
- NumPy FP32 reference and native C++ integer verification; scripts extract latency, interval and DSP/LUT/FF/BRAM from HLS synthesis reports.

| Numerical metric | Recorded result |
|---|---:|
| Integer-reference mismatches | 0 / 676 |
| MAE vs FP32 | 0.00385018 |
| RMSE vs FP32 | 0.00485245 |
| Maximum absolute error | 0.01659325 |
| INT32 accumulator overflows | 0 |

The course-level CNN comprises Conv2D/ReLU, max pooling, Dense(100)/ReLU, Dropout and Dense/Softmax. Its owner-reported results are **accuracy >80%** and **core II ≤676 cycles**, with a **100 MHz target**. The complete network source, trained weights and raw historical synthesis reports are unavailable in this repository. These figures are contextual results, not measurements reproduced by the current operator testbench. No numerical FPGA resource savings or cross-device speedup is claimed. See [HLS results provenance](docs/RESULTS.md).

## CUDA path

- Plain single-thread C++ GEMM reference, without BLAS.
- Naive CUDA: one thread per output element, global-memory input reads.
- Tiled CUDA: cooperative loads into shared memory, block synchronization and data reuse; tile sizes 8×8, 16×16 and 32×32.
- Boundary checks handle partial tiles. Correctness uses an absolute-plus-relative tolerance of `1e-5 + 1e-5 * abs(reference)`.
- All tested GPU configurations passed numerical checks against the FP32 CPU reference.

### Benchmark environment and method

NVIDIA GeForce RTX 2060 (6 GB, Windows WDDM), driver 551.76, CUDA Toolkit 12.4, MSVC 19.39, `sm_75`, and Nsight Compute 2024.1. Build uses `-O2`, C++17 and UTF-8.

Each implementation uses 10 warmups and 100 samples. CPU uses a host clock (1000 inner iterations per sample for the tiny case); GPU uses CUDA Events around individual kernel launches. Throughput is `2*M*N*K / time`. GPU event measurements can include device timeline gaps and are distinct from Nsight kernel duration.

**Timings exclude allocation, Im2col and host/device transfers. They do not represent end-to-end CNN latency.** CPU speedups compare only with the plain single-thread reference, not optimized BLAS.

### Standalone benchmark results

| Workload | Implementation | Mean (ms) | Median (ms) | GFLOPS | Speedup vs CPU |
|---|---|---:|---:|---:|---:|
| Original | CPU | 0.004262 | 0.004242 | 2.855 | 1.000× |
| Original | Naive | 0.026094 | 0.022400 | 0.466 | 0.163× |
| Original | Tiled 8 | 0.012580 | 0.009904 | 0.967 | 0.339× |
| Original | Tiled 16 | 0.013678 | 0.010240 | 0.890 | 0.312× |
| Original | Tiled 32 | 0.015175 | 0.012480 | 0.802 | 0.281× |
| Expanded | CPU | 4.785097 | 4.725400 | 2.604 | 1.000× |
| Expanded | Naive | 0.046948 | 0.046864 | 265.400 | 101.923× |
| Expanded | Tiled 8 | 0.053134 | 0.053120 | 234.500 | 90.056× |
| Expanded | Tiled 16 | 0.039254 | 0.038800 | 317.423 | 121.902× |
| Expanded | Tiled 32 | 0.048067 | 0.047968 | 259.221 | 99.550× |

The original workload favors CPU execution. For the expanded workload, tile16 reduces mean latency by **16.4%** relative to naive CUDA (**1.196× speedup**). Larger tiles do not automatically improve performance.

### Nsight Compute: expanded workload

| Metric | Naive | Tiled 16 |
|---|---:|---:|
| Kernel duration | 42.14 μs | 33.44 μs |
| Registers/thread | 52 | 39 |
| Static shared memory/block | 0 | 2048 bytes |
| Theoretical occupancy | 100% | 100% |
| Achieved occupancy | 77.67% | 78.80% |
| DRAM throughput utilization | 3.85% | 4.85% |
| Blocks × threads/block | 172 × 256 | 172 × 256 |

These individual profiling captures show a 20.6% duration reduction, separately from the repeated standalone benchmark. Shared-memory tiling enables reuse, but these sections alone do not quantify reductions in total memory traffic. Occupancy changes only slightly, and neither kernel saturates DRAM bandwidth in these captures. Profiling replay changes program timing: do not use benchmark CSVs produced under `ncu` as standalone latency results.

FPGA and GPU differ in precision, execution model and available measurements. **This is an architectural comparison rather than a strict apples-to-apples performance benchmark.** II is not transaction latency.

## Repository layout

- `hls/`: FP32/INT8 HLS convolution implementations.
- `reference/`, `testbench/`: numerical references and native C++ verification.
- `scripts/`: vector generation, verification, HLS synthesis and report parsing.
- `cuda/`: CPU baseline, naive/tiled kernels and unified benchmark.
- `results/`: numerical artifacts and standalone CUDA benchmark outputs.
- `results/profiling/`: Nsight text reports.
- `docs/`: HLS result provenance and supporting instructions.

## Run HLS numerical verification

Requires Python 3.9+, NumPy and a C++17 compiler. Run from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/generate_test_data.py
g++ -std=c++17 -O2 hls/conv_top.cpp testbench/tb_conv.cpp -o conv_tb
./conv_tb results/vectors.txt results/int8_output.txt
python scripts/verify.py results/int8_output.txt
```

Native verification is not vendor C simulation or RTL co-simulation. For synthesis, set `HLS_PART` to a supported installed FPGA part, then run:

```bash
vitis_hls -f scripts/run_hls.tcl
python scripts/parse_hls_report.py build_conv_fp32/solution1/syn/report/conv_fp32_csynth.xml build_conv_int8/solution1/syn/report/conv_int8_csynth.xml
```

The synthesis script uses a 10 ns clock constraint. Inspect actual timing, device and resource reports before drawing hardware conclusions.

## Run CUDA benchmark (Windows CMD)

Use an x64 Visual Studio developer prompt with the CUDA-12.4-compatible MSVC 19.39 toolset. The two kernel source files must be next to `benchmark.cu`; the benchmark includes them directly.

```bat
if not exist results mkdir results
nvcc -O2 -std=c++17 -arch=sm_75 -Xcompiler "/utf-8" cuda\benchmark.cu -o benchmark.exe
benchmark.exe
```

Outputs: `results/cuda_benchmark.csv` and `results/benchmark_samples.csv`. A successful run ends with `ALL CORRECTNESS CHECKS PASSED`.

For profiling, use an administrator CMD if GPU performance counter access is restricted. Keep profiler-generated benchmark CSVs separate:

```bat
if not exist results\profiling mkdir results\profiling
cd results\profiling
if not exist results mkdir results
ncu --section SpeedOfLight --section LaunchStats --section Occupancy --launch-skip 455 --launch-count 1 -o expanded_naive_admin ..\..\benchmark.exe
ncu --section SpeedOfLight --section LaunchStats --section Occupancy --launch-skip 677 --launch-count 1 -o expanded_tiled16_admin ..\..\benchmark.exe
ncu --import expanded_naive_admin.ncu-rep --page details > expanded_naive_details.txt
ncu --import expanded_tiled16_admin.ncu-rep --page details > expanded_tiled16_details.txt
```

Launch offsets assume the current benchmark order and 111 launches per GPU implementation (one correctness check, ten warmups, 100 timed launches). Recalculate offsets if the harness changes; verify the captured kernel name and grid dimensions.
