# Docker Setup

This repository includes a Docker environment for running CONTRA without relying on the host Python installation. Source code is copied into `/workspace`. Datasets, checkpoints, logs, and run outputs should be mounted from the host.

## Versions

- Python base image: `python:3.11.13-slim-bookworm`
- PyTorch wheels: `torch==2.6.0`, `torchvision==0.21.0`
- Python packaging tools: `pip==25.2`, `setuptools==80.9.0`, `wheel==0.45.1`
- Non-PyTorch dependencies: see `requirements-docker.txt`

The default image runs on CPU. GPU runs need a CUDA-compatible image and host driver.

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

## Checks

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

## Run The Experiment Pipeline

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
  --seed 42
```

For later incremental steps, run CONTRA refinement/adaptation directly. The command loads the previous `worker_restore` checkpoint and does not need a current-stage raw-worker checkpoint:

```bash
python run_contra.py \
  --step 1 \
  --model cifar-resnet18 \
  --dataset cifar-10 \
  --noise_ratio 0.2 \
  --noise_type symmetric \
  --balanced \
  --num_epochs 50 \
  --adapt_epochs 5 \
  --adapt_iter_num 3 \
  --learning_rate 0.05 \
  --optimizer adam \
  --batch_size 128 \
  --seed 42
```

Use these scripts for repeated seeds or sweeps:

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

The image excludes large experiment outputs:

- `data/`
- `ckpt/`
- `logs/`
- `runs/`

Mount these directories from the host whenever you run experiments.

## GPU Variant

For GPU-optimized runs, use a CUDA-compatible PyTorch base image or replace the PyTorch install line with the matching CUDA wheel index for your driver and CUDA runtime. Then launch with:

```bash
docker run --gpus all --rm -it \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/ckpt:/workspace/ckpt" \
  -v "$PWD/logs:/workspace/logs" \
  -v "$PWD/runs:/workspace/runs" \
  contra-repro:gpu bash
```

For repeatable GPU runs, record the host GPU model, driver version, CUDA runtime, command-line arguments, seed, and checkpoint path with each result.

## Troubleshooting

- If the build is slow, the PyTorch wheels are usually the largest download.
- If Docker cannot see local files on Windows, check Docker Desktop file sharing permissions and use the PowerShell command form above.
- If generated datasets are missing inside the container, confirm that the host `data/` directory is mounted to `/workspace/data`.
- If you need bit-for-bit stronger reproducibility, pin the base image by digest after choosing the exact Docker registry image you want to archive.
