# Results and provenance

## Original project

Source: project owner's account of completed ECEN226 work. Recorded results: Im2col and GEMM HLS convolution; core II ≤676 cycles; original throughput target achieved under a 100 MHz target; full CNN classification accuracy >80%. Original source, weights and synthesis reports are unavailable in this release. Accuracy belongs to the full CNN and is not an operator error metric. Clock target is not proof of achieved post-route timing.

## Owner's additional experiments

The owner reports that new experiments are complete and their outcomes match earlier results. No additional raw numerical output, synthesis reports, FPGA part, tool version or resource counts were supplied for this release. This statement is recorded in `results/project_results.json`; it is not converted into fabricated tool output or per-version measurements.

## Included numerical artifact

`results/numerical_results.json` preserves the earlier native C++ check, using seed 42 and synthetic inputs/weights. MAE, RMSE and maximum error compare dequantized INT32 output with NumPy FP32 convolution. The integer mismatch count compares against an independent integer reference. These synthetic weights are not recovered trained CNN weights.

A legacy `saturation_count` value of 4 actually counted quantized values at magnitude 127, including legitimate values that had not been clipped. It has been renamed `legacy_abs127_count`; actual clipping count in the retained result is null. The script now counts rounded values outside [-128,127] before clipping. No experimental results were regenerated for packaging. Accumulator overflow counts are computed with the integer reference; there is no synthesizable overflow counter.

## Synthesis

`results/synthesis_results.json` records reports as unavailable in this package. A provided report parser can populate real tool measurements later. Latency and interval are different metrics; the historical II threshold cannot establish the new top-level throughput. Resource reduction percentages cannot be calculated without paired reports.
