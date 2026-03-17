# EdgeCompression-Benchmark-2026

Unified benchmark of classical vs. neural compression codecs on consumer Blackwell GPU (RTX 5090) with energy profiling and adaptive Pareto-optimal codec selector.

## Overview

This repository contains the code and results for a systematic comparison of classical and neural compression methods across three domains:

- **Images**: JPEG, H.265/HEVC, Ballé2018, Cheng2020
- **Audio**: AAC, Opus, EnCodec
- **Video**: H.264, H.265, AV1

Metrics: PSNR, MS-SSIM, LPIPS, PESQ, STOI, SI-SDR, throughput (fps), energy (Joules via Zeus/NVML).

## Hardware

- GPU: NVIDIA GeForce RTX 5090 (Blackwell, 32GB VRAM)
- CPU: AMD Ryzen 7 9800X3D
- Energy measurement: Zeus (GPU via NVML), RAPL (CPU, bare metal only)

## Adaptive Selector

The core contribution is an empirical Pareto-optimal codec selector:
```bash
python src/selector/cli.py --domain image --max-bpp 0.3 --max-lpips 0.3
python src/selector/cli.py --domain audio --max-bpp 12
python src/selector/cli.py --domain video --max-enc-ms 10 --show-pareto
```

## Setup
```bash
conda create -n tesi python=3.11
conda activate tesi
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install compressai encodec zeus-ml lpips pytorch-msssim pesq pystoi librosa soundfile pandas matplotlib tqdm opencv-python ffmpeg-python av black
```

## Structure
```
src/
  benchmark/      # Benchmark scripts per domain
  selector/       # Adaptive selector + CLI
  utils/          # Energy monitor (Zeus)
results/          # CSV results (gitignored, available on request)
notebooks/        # Plotting scripts
```

## Citation

If you use this benchmark, please cite the thesis (BibTeX forthcoming).
