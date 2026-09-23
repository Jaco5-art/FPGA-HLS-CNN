#pragma once
#ifdef __SYNTHESIS__
#include <ap_int.h>
#else
#include <cstdint>
using int8_hls = int8_t;
using int32_hls = int32_t;
#endif
#ifdef __SYNTHESIS__
using int8_hls = ap_int<8>;
using int32_hls = ap_int<32>;
#endif
constexpr int H=28, K=3, O=26, P=O*O, C=K*K;
void conv_fp32(const float input[H*H], const float weight[C], float output[P]);
void conv_int8(const int8_hls input[H*H], const int8_hls weight[C], int32_hls output[P]);
