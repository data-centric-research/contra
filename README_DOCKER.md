# Docker Reproducibility Guide

This repository includes a pinned CPU Docker environment for reproducing the released CONTRA code without relying on a host Python installation. The image keeps source code inside `/workspace` and expects datasets, checkpoints, logs, and run outputs to be mounted from the host.

## What Is Pinned

- Python base image: `python:3.10.14-slim-bookworm`
- PyTorch CPU wheels: `torch==2.9.1+cpu`, `torchvision==0.24.1+cpu`
- Python packaging tools: `pip==25.2`, `setuptools==80.9.0`, `wheel==0.45.1`
- Non-PyTorch dependencies: see `requirements-docker.txt`

The Docker image is CPU-first for portability. GPU runs are possible, but CUDA images and drivers vary by machine, so they are documented separately below.

## Prerequisites

- Docker Engine or Docker Desktop
- Optional: Docker Compose v2
- Optional for GPU runs: NVIDIA Container Toolkit and a CUDA-compatible host driver

## Build The Image

From the repository root:

```bash
docker build -t contra-repro:cpu .
```

Or with Docker Compose:

```bash
docker compose build
```

## Quick Sanity Checks

```bash
docker run --rm contra-repro:cpu python -c "import torch, torchvision, numpy, sklearn; print(torch.__version__, torchvision.__version__)"
docker run --rm contra-repro:cpu python run_contra.py --help
docker run --rm contra-repro:cpu python scripts/evaluate_checkpoint.py --help
```

## Start An Interactive Container

On Linux or macOS:

```bash
docker run --rm -it \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/ckpt:/workspace/ckpt" \
  -v "$PWD/logs:/workspace/logs" \
  -v "$PWD/runs:/workspace/runs" \
  contra-repro:cpu bash
```

On Windows PowerShell:

```powershell
docker run --rm -it `
  -v ${PWD}\data:/workspace/data `
  -v ${PWD}\ckpt:/workspace/ckpt `
  -v ${PWD}\logs:/workspace/logs `
  -v ${PWD}\runs:/workspace/runs `
  contra-repro:cpu bash
```

With Docker Compose:

```bash
docker compose run --rm contra bash
```

## Run The Paper-Style Pipeline

Inside the container, generate staged tensor files first. Example for CIFAR-10 with symmetric 20% label noise:

```bash
python gen_dataset/gen_cifar10_exp_data.py \
  --data_dir ./data/cifar-10/normal \
  --noise_type symmetric \
  --noise_ratio 0.2 \
  --num_versions 4 \
  --retention_ratios 0.5 0.3 0.1 0.05 \
  --balanced
```

Then run step 0 training:

```bash
python run_experiment.py \
  --step 0 \
  --model cifar-resnet18 \
  --dataset cifar-10 \
  --noise_ratio 0.2 \
  --noise_type symmetric \
  --balanced \
  --num_epochs 50 \
  --learning_rate 0.05 \
  --optimizer adam \
  --batch_size 128 \
  --seed 1
```

For later incremental steps, train the raw worker first and then run CONTRA refinement/adaptation:

```bash
python run_experiment.py \
  --step 1 \
  --model cifar-resnet18 \
  --dataset cifar-10 \
  --noise_ratio 0.2 \
  --noise_type symmetric \
  --balanced \
  --num_epochs 50 \
  --learning_rate 0.05 \
  --optimizer adam \
  --batch_size 128 \
  --seed 1

python run_contra.py \
  --step 1 \
  --model cifar-resnet18 \
  --dataset cifar-10 \
  --noise_ratio 0.2 \
  --noise_type symmetric \
  --balanced \
  --num_epochs 50 \
  --learning_rate 0.05 \
  --optimizer adam \
  --batch_size 128 \
  --seed 1
```

Use the helper scripts for repeated seeds or sweeps:

```bash
python scripts/run_multi_seed.py --help
python scripts/run_sweep.py --help
```

Evaluate classification accuracy and retrieval mAP from a saved checkpoint:

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint_path ./ckpt/path/to/checkpoint.pth \
  --dataset cifar-10 \
  --model cifar-resnet18 \
  --noise_ratio 0.2 \
  --noise_type symmetric \
  --balanced \
  --step 1 \
  --batch_size 128 \
  --eval_map
```

## Data And Output Volumes

The image intentionally excludes heavy or generated artifacts:

- `data/`
- `ckpt/`
- `logs/`
- `runs/`

Mount these directories from the host whenever you run experiments. This keeps the image small and makes it clear which files are code versus local experimental artifacts.

## GPU Variant

The default Dockerfile installs CPU-only PyTorch wheels. For GPU runs, use a CUDA-compatible PyTorch base image or replace the PyTorch install line with the matching CUDA wheel index for your driver and CUDA runtime. Then launch with:

```bash
docker run --gpus all --rm -it \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/ckpt:/workspace/ckpt" \
  -v "$PWD/logs:/workspace/logs" \
  -v "$PWD/runs:/workspace/runs" \
  contra-repro:gpu bash
```

For exact publication-grade reruns, record the host GPU model, driver version, CUDA runtime, command-line arguments, seed, and checkpoint path with each result.

## Troubleshooting

- If the build is slow, the PyTorch CPU wheels are usually the largest download.
- If Docker cannot see local files on Windows, check Docker Desktop file sharing permissions and use the PowerShell command form above.
- If generated datasets are missing inside the container, confirm that the host `data/` directory is mounted to `/workspace/data`.
- If you need bit-for-bit stronger reproducibility, pin the base image by digest after choosing the exact Docker registry image you want to archive.
