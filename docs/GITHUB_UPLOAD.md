# 上传到 GitHub

1. 解压 `fpga-hls-cnn.zip`。
2. 新建空仓库，建议名称 `fpga-hls-cnn`。
3. 上传解压后 `fpga-hls-cnn` 文件夹里的内容，让 `README.md` 位于仓库根目录。不要仅上传压缩包。

也可以在解压后的项目目录执行（将最后的仓库地址换成你自己的）：

```bash
git init
git add .
git commit -m "Add FPGA HLS CNN project with INT8 convolution extension"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fpga-hls-cnn.git
git push -u origin main
```

仓库简介可用：

FPGA/HLS CNN acceleration with GEMM-based convolution, INT8 quantization, INT32 accumulation, and numerical verification.

建议 Topics：`fpga`、`hls`、`cnn`、`int8`、`quantization`、`edge-ai`、`gemm`。
