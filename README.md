# FPGA/HLS CNN Inference Accelerator with INT8 Quantization

**FPGA/HLS CNN 推理加速与定点量化优化**

A CNN acceleration project originating from the Santa Clara University ECEN226 final exam, extended with an independently implemented INT8 convolution kernel. The project combines GEMM-based convolution, HLS operator design, explicit quantization and numerical verification.

## Project overview

The original CNN used a 28×28×1 input, Conv2D with one 3×3 filter and ReLU, 2×2 max pooling (stride 2), Flatten, Dense(100, ReLU), Dropout, and a final Dense + Softmax layer. The convolution was expressed as `Im2col()` followed by `GemmConv2d0()`, with notebook trace data used for verification.

The current source release provides the convolution extension: a matching FP32 baseline and an INT8 implementation using the same im2col + GEMM structure. It preserves the operator design of the project while introducing explicit data types and quantization scales.

## Results

| Metric | Recorded result | Scope / source |
|---|---|---|
| Core initiation interval | ≤676 cycles | Original project, owner-reported |
| Clock target | 100 MHz | Original design target; current synthesis script uses 10 ns |
| Throughput | Original prescribed target achieved | Owner-reported; numerical throughput value not retained |
| Full CNN classification accuracy | >80% | Original full-network result, owner-reported |
| Extension MAE vs FP32 | 0.00385018 | Included native C++ numerical check |
| Extension RMSE vs FP32 | 0.00485245 | Included native C++ numerical check |
| Extension maximum absolute error | 0.01659325 | Included native C++ numerical check |
| Integer reference mismatches | 0 / 676 outputs | Included native C++ numerical check |
| INT32 accumulator overflows | 0 | Included reference calculation |

The owner reports completing additional experiments with outcomes unchanged from the earlier project. This release retains the earlier recorded figures with their original scope. New raw reports and per-version DSP/LUT/FF/BRAM values were not supplied, so this repository does not infer numerical resource savings or a new kernel II from the historical values. See [results provenance](docs/RESULTS.md).

## Quantized convolution

| Component | Design |
|---|---|
| Input / kernel | 28×28×1 / one 3×3 kernel |
| Operation | Valid cross-correlation, stride 1, no bias or activation |
| Output shape | 26×26 |
| Input and weights | Signed INT8 |
| Accumulator and output | Signed INT32 |
| Input / weight scale | 1/127 each |
| Output dequantization scale | 1/16129 |
| Quantizer | Nearest-even rounding followed by clipping to [-128, 127] |

INT32 output preserves the accumulated convolution before dequantization; INT8/INT16 output requantization is not included. Nine INT8 products have a maximum absolute sum bounded by 147456, within signed INT32 range.

`Im2col()` produces a 676×9 matrix. `GemmConv2d0()` computes 676 dot products of length 9. The inner loops request `PIPELINE II=1` to expose sequential work to the HLS scheduler. Achieved loop II and top-level transaction interval depend on synthesis; the directive alone guarantees neither. FP32 recurrence latency may limit its inner-loop II. No partition or unroll directives are added.

## Repository contents

- `hls/`: FP32 and INT8 convolution source and type declarations.
- `reference/`: NumPy FP32 operator reference.
- `testbench/`: native C++ INT8 testbench.
- `scripts/`: deterministic data generation, numerical verification, HLS synthesis and XML report extraction.
- `results/`: retained input/weight arrays, output, quantization configuration and recorded results.
- `docs/`: project history, results provenance and GitHub upload instructions.

The original complete CNN source, trained weights and synthesis reports are unavailable. The original course document is retained by the owner but was not supplied for this package. This source release therefore reproduces the extension, not the historical complete network.

## Local numerical workflow

Python 3.9+, NumPy and a C++17 compiler are needed. From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/generate_test_data.py
g++ -std=c++17 -O2 hls/conv_top.cpp testbench/tb_conv.cpp -o conv_tb
./conv_tb results/vectors.txt results/int8_output.txt
python scripts/verify.py results/int8_output.txt
```

On Windows, compile with a C++17 toolchain and invoke the resulting `.exe`. These commands regenerate the sample results; packaging this release did not rerun experiments. The FP32 HLS top is included for synthesis, while the provided numerical testbench exercises the INT8 top. Native execution uses standard integer types; HLS synthesis selects AMD `ap_int` types. Native verification is not vendor C simulation or RTL co-simulation.

## HLS synthesis workflow

On a machine with Vitis HLS and a supported FPGA part, set `HLS_PART` to the installed part identifier, then run from this directory:

```bash
vitis_hls -f scripts/run_hls.tcl
python scripts/parse_hls_report.py build_conv_fp32/solution1/syn/report/conv_fp32_csynth.xml build_conv_int8/solution1/syn/report/conv_int8_csynth.xml
```

The script applies the same FPGA part and 10 ns clock constraint to both versions. Inspect latency, top-level interval, estimated clock and DSP/LUT/FF/BRAM in the generated report. Missing XML fields remain null. Check the report device, clock, tool version and achieved timing before interpreting comparisons. The scripts currently perform synthesis only, not vendor C simulation or RTL co-simulation.

## Upload

See [GitHub upload guide](docs/GITHUB_UPLOAD.md). Suggested repository name: `fpga-hls-cnn`. No license has been selected; choose one appropriate to your ownership and course policy before granting reuse rights.
