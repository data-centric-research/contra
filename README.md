# CONTRA: A Continual Train, Refinement and Adaptation Framework for Building Robust Web Image Recognition Systems

This repository contains the public code for the ICANN 2026 paper:
**CONTRA: A Continual Train, Refinement and Adaptation Framework for Building Robust Web Image Recognition Systems**.

CONTRA combines a fixed rehearsal buffer, teacher-student noisy-label refinement with spectral normalization on the teacher, and request-time adaptation. The released scripts directly cover the CIFAR-10, CIFAR-100, and Oxford-IIIT Pet experiments. The compact Food-101 and WebVision checks reported in the paper use the same tensor-file interface after local preprocessing.

## Repository Layout

```text
args_paser.py                 Shared command-line parser
run_experiment.py             Raw and incremental training entry point
core_model/                   CONTRA training, refinement, and adaptation code
gen_dataset/                  Dataset split and noisy incremental data generation
configs/                      Dataset paths, class names, and class mappings
baseline_code/colearn/        LNL baselines: Co-teaching, Co-teaching+, JoCoR, DivideMix
baseline_code/cotta-main/     CoTTA baseline adapter
baseline_code/PLF-main/       PLF baseline adapter
baseline_code/sotta/          SoTTA baseline adapter
```

## Environment

Python 3.10 is recommended. A local virtual environment or Conda environment is sufficient; this repository does not require a Dockerfile.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset Generation

Generate the staged tensor files before training. The scripts write to:

```text
data/<dataset>/gen/<case>/
```

where `<case>` has the form:

```text
nr_<noise_ratio>_nt_<noise_type>_balanced
```

when `--balanced` is used. The paper uses a 50% clean initial split, a fixed 10% rehearsal buffer sampled from that initial split, four incremental stages, and 20% label noise unless otherwise stated.

### CIFAR-10, Symmetric Noise

```bash
PYTHONPATH=. python gen_dataset/gen_cifar10_exp_data.py \
  --data_dir ./data/cifar-10/normal \
  --noise_type symmetric \
  --noise_ratio 0.2 \
  --num_versions 4 \
  --retention_ratios 0.5 0.3 0.1 0.05 \
  --balanced
```

### CIFAR-100, Asymmetric Noise

```bash
PYTHONPATH=. python gen_dataset/gen_cifar100_exp_data.py \
  --data_dir ./data/cifar-100/normal \
  --noise_type asymmetric \
  --noise_ratio 0.2 \
  --num_versions 4 \
  --retention_ratios 0.5 0.3 0.1 0.05 \
  --balanced
```

### Oxford-IIIT Pet, Symmetric Noise

```bash
PYTHONPATH=. python gen_dataset/gen_pet37_exp_data.py \
  --data_dir ./data/pet-37/normal \
  --noise_type symmetric \
  --noise_ratio 0.2 \
  --num_versions 4 \
  --retention_ratios 0.5 0.3 0.1 0.05 \
  --balanced
```

After generation, a case directory contains files such as:

```text
data/cifar-10/gen/nr_0.2_nt_symmetric_balanced/
  aux_data.npy
  aux_label.npy
  test_data.npy
  test_label.npy
  step_0/train_data.npy
  step_0/train_label.npy
  step_1/train_data.npy
  step_1/train_label.npy
  step_2/train_data.npy
  step_2/train_label.npy
  step_3/train_data.npy
  step_3/train_label.npy
  step_4/train_data.npy
  step_4/train_label.npy
```

## Paper-Aligned Backbones

| Dataset | Backbone in the paper | `--model` value |
| --- | --- | --- |
| CIFAR-10 | ResNet-18 | `cifar-resnet18` |
| CIFAR-100 | WideResNet-40-2 | `cifar-wideresnet40` |
| Oxford-IIIT Pet | WideResNet-50-2 | `wideresnet50` |

## Training

Run step 0 before later incremental stages because each stage loads the previous checkpoint from `ckpt/`.

### CIFAR-10 Step 0

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py \
  --step 0 \
  --model cifar-resnet18 \
  --dataset cifar-10 \
  --noise_ratio 0.2 \
  --noise_type symmetric \
  --balanced \
  --num_epochs 50 \
  --learning_rate 0.05 \
  --optimizer adam \
  --batch_size 128
```

### CIFAR-10 Step 1

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py \
  --step 1 \
  --model cifar-resnet18 \
  --dataset cifar-10 \
  --noise_ratio 0.2 \
  --noise_type symmetric \
  --balanced \
  --num_epochs 50 \
  --learning_rate 0.05 \
  --optimizer adam \
  --batch_size 128
```

Repeat with `--step 2`, `--step 3`, and `--step 4`. Use `--dataset pet-37 --model wideresnet50` for Oxford-IIIT Pet and `--dataset cifar-100 --model cifar-wideresnet40 --noise_type asymmetric` for the CIFAR-100 retrieval-style setting.

## Baselines

The paper separates source-free continual TTA baselines from source-available adaptation methods. The following local adapters are included for reproducibility checks.

### LNL Baselines

Co-teaching, Co-teaching+, JoCoR, and DivideMix share the entry point under `baseline_code/colearn/`.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python baseline_code/colearn/main.py \
  --step 1 \
  --model cifar-resnet18 \
  --dataset cifar-10 \
  --noise_ratio 0.2 \
  --noise_type symmetric \
  --balanced \
  --num_epochs 15 \
  --batch_size 128 \
  --uni_name DivideMix
```

### TTA Baselines

CoTTA, PLF, and SoTTA are provided as local adapters under `baseline_code/cotta-main/`, `baseline_code/PLF-main/`, and `baseline_code/sotta/`. They expect the same generated tensor files and a compatible stage-0 checkpoint.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python baseline_code/sotta/run_sotta.py \
  --step 1 \
  --model cifar-resnet18 \
  --dataset cifar-10 \
  --noise_ratio 0.2 \
  --noise_type symmetric \
  --balanced \
  --batch_size 128 \
  --uni_name SoTTA
```

## Notes on Food-101 and WebVision

The paper reports compact Food-101 and WebVision checks under the same staged tensor protocol. Public scripts include `food-101` in the shared data interface, but the raw Food-101/WebVision preprocessing used for the compact checks is not included here. To reproduce those checks, preprocess the datasets into the same `train_data.npy`, `train_label.npy`, `test_data.npy`, `test_label.npy`, and staged `step_<k>/train_*.npy` layout.

## License

This repository is released under the MIT license.
