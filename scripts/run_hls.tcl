# Usage: vitis_hls -f scripts/run_hls.tcl ; set HLS_PART to an installed FPGA part.
if {![info exists ::env(HLS_PART)]} { error "Set HLS_PART to your installed FPGA part" }
foreach top {conv_fp32 conv_int8} {
  open_project -reset "build_$top"
  set_top $top
  add_files hls/conv_top.cpp -cflags "-D__SYNTHESIS__"
  open_solution -reset solution1
  set_part $::env(HLS_PART)
  create_clock -period 10 -name default
  csynth_design
  close_project
}
