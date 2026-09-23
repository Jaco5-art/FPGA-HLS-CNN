#include "conv.hpp"
// One output channel; valid cross-correlation, matching typical Conv2D semantics.
template<class T> void Im2col(const T input[H*H], T col[P*C]) {
  for (int p=0; p<P; ++p) {
    for (int k=0; k<C; ++k) {
#pragma HLS PIPELINE II=1
      col[p*C+k]=input[(p/O+k/K)*H+(p%O+k%K)];
    }
  }
}
template<class T, class A> void GemmConv2d0(const T col[P*C], const T weight[C], A output[P]) {
  for (int p=0; p<P; ++p) {
    A sum=0;
    for (int k=0; k<C; ++k) {
#pragma HLS PIPELINE II=1
      sum += A(col[p*C+k])*A(weight[k]);
    }
    output[p]=sum;
  }
}
void conv_fp32(const float input[H*H], const float weight[C], float output[P]) {
  float col[P*C];
  Im2col(input,col);
  GemmConv2d0(col,weight,output);
}
void conv_int8(const int8_hls input[H*H], const int8_hls weight[C], int32_hls output[P]) {
  int8_hls col[P*C];
  Im2col(input,col);
  GemmConv2d0(col,weight,output);
}
