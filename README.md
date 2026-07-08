# CONTRA: A Continual Train, Refinement and Adaptation Framework for Building Robust Web Image Recognition Systems

This repository contains the public code for the ICANN 2026 paper:
**CONTRA: A Continual Train, Refinement and Adaptation Framework for Building Robust Web Image Recognition Systems**.

CONTRA combines a fixed rehearsal buffer, teacher-student noisy-label refinement with spectral normalization on the teacher, and request-time adaptation. The released scripts directly cover the CIFAR-10, CIFAR-100, and Oxford-IIIT Pet experiments. The compact Food-101 and WebVision checks reported in the paper use the same tensor-file interface after local preprocessing.

## Repository Layout

```text
args_paser.py                 Shared command-line parser
run_experiment.py             Raw and incremental training entry point
run_contra.py                 CONTRA refinement and request-time adaptation entry point
core_model/                   CONTRA training, refinement, and adaptation code
gen_dataset/                  Dataset split and noisy incremental data generation
configs/                      Dataset paths, class names, and class mappings
baseline_code/colearn/        LNL baselines: Co-teaching, Co-teaching+, JoCoR, DivideMix
baseline_code/cotta-main/     CoTTA baseline adapter
baseline_code/PLF-main/       PLF baseline adapter
baseline_code/sotta/          SoTTA baseline adapter
Dockerfile                    Reproducible CPU container environment
docker-compose.yml            Optional mounted-volume Docker Compose entry point
README_DOCKER.md              Docker build, run, and troubleshooting guide
```

## Environment

Python 3.11 is recommended. For a local virtual environment or Conda environment:

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

For a reproducible CPU Docker environment, see [README_DOCKER.md](README_DOCKER.md):

```bash
docker build -t contra-repro:cpu .
docker run --rm contra-repro:cpu python run_contra.py --help
```

On Windows PowerShell, replace command prefixes such as `PYTHONPATH=.` and `CUDA_VISIBLE_DEVICES=0` with:

```powershell
$env:PYTHONPATH='.'
$env:CUDA_VISIBLE_DEVICES='0'
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

For the Pet retrieval table under asymmetric noise, regenerate the Pet tensors with `--noise_type asymmetric` and use `--eval_map` during evaluation:

```bash
PYTHONPATH=. python gen_dataset/gen_pet37_exp_data.py \
  --data_dir ./data/pet-37/normal \
  --noise_type asymmetric \
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

## Backbones Used in the Paper

| Dataset | Backbone in the paper | `--model` value |
| --- | --- | --- |
| CIFAR-10 | ResNet-18 | `cifar-resnet18` |
| CIFAR-100 | WideResNet-40-2 | `cifar-wideresnet40` |
| Oxford-IIIT Pet | WideResNet-50-2 | `wideresnet50` |

## Training and CONTRA Adaptation

Run step 0 before later incremental stages because each CONTRA stage loads the previous `worker_restore` checkpoint from `ckpt/`. For stages 1--4, run CONTRA refinement and request-time adaptation directly; the raw worker commands are reserved for the Raw and Rehearsal baselines.

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

### CIFAR-10 Step 1 CONTRA

```bash
CUDA_VISIBLE_DEVICES=0 python run_contra.py \
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
  --batch_size 128
```

Add `--eval_map` to report retrieval mAP from penultimate-layer cosine rankings.

Repeat the CONTRA command with `--step 2`, `--step 3`, and `--step 4`. Use `--dataset pet-37 --model wideresnet50` for Oxford-IIIT Pet and `--dataset cifar-100 --model cifar-wideresnet40 --noise_type asymmetric --eval_map` for the CIFAR-100 retrieval-style setting.

For `run_experiment.py` training runs, add `--use_early_stopping` to split the training data into train/validation subsets and activate the `--early_stopping_patience`, `--early_stopping_accuracy_threshold`, and `--validation_ratio` controls.

## Baselines

The paper separates source-free continual TTA baselines from source-available adaptation methods. The following local adapters are included for reproducibility checks.

### Source-Available Rehearsal

The Rehearsal baseline reuses the same fixed auxiliary buffer. First create the stage raw worker with a baseline-specific `--uni_name`, then fine-tune that worker on `aux_data.npy`.

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
  --batch_size 128 \
  --model_suffix worker_raw \
  --uni_name Rehearsal

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
  --batch_size 128 \
  --train_aux \
  --uni_name Rehearsal
```

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

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python baseline_code/cotta-main/cifar/tta.py \
  --step 1 \
  --model cifar-resnet18 \
  --dataset cifar-10 \
  --noise_ratio 0.2 \
  --noise_type symmetric \
  --balanced \
  --batch_size 128 \
  --uni_name CoTTA

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python baseline_code/PLF-main/cifar/run_plf.py \
  --step 1 \
  --model cifar-resnet18 \
  --dataset cifar-10 \
  --noise_ratio 0.2 \
  --noise_type symmetric \
  --balanced \
  --batch_size 128 \
  --uni_name PLF
```

## Notes on Food-101 and WebVision

The paper reports compact Food-101 and WebVision checks under the same staged tensor protocol. Use the shared ImageFolder/list-file preprocessing script to convert raw images into the same `.npy` layout used by CIFAR and Pet.

ImageFolder-style directories:

```bash
PYTHONPATH=. python gen_dataset/gen_webvision_exp_data.py \
  --dataset webvision \
  --train_root ./data/webvision/raw/train \
  --test_root ./data/webvision/raw/val \
  --num_classes 100 \
  --max_train_per_class 500 \
  --max_test_per_class 100 \
  --noise_type symmetric \
  --noise_ratio 0.2 \
  --balanced
```

WebVision-style list files are also supported. Each line may be `relative/path.jpg label` or `relative/path.jpg,label`.

```bash
PYTHONPATH=. python gen_dataset/gen_webvision_exp_data.py \
  --dataset webvision \
  --image_root ./data/webvision/raw \
  --train_list ./data/webvision/train_filelist.txt \
  --test_list ./data/webvision/val_filelist.txt \
  --num_classes 100 \
  --max_train_per_class 500 \
  --max_test_per_class 100 \
  --noise_type symmetric \
  --noise_ratio 0.2 \
  --balanced
```

For Food-101, use the same script with `--dataset food-101`, or point it at class-folder train/test directories. For compact WebVision, pass the same `--num_classes 100` option to every training, CONTRA, and evaluation command so that the classifier head matches the generated subset.

```bash
PYTHONPATH=. python run_experiment.py \
  --step 0 \
  --model wideresnet50 \
  --dataset webvision \
  --num_classes 100 \
  --noise_ratio 0.2 \
  --noise_type symmetric \
  --balanced \
  --num_epochs 50 \
  --learning_rate 0.05 \
  --optimizer adam \
  --batch_size 128

PYTHONPATH=. python run_contra.py \
  --step 1 \
  --model wideresnet50 \
  --dataset webvision \
  --num_classes 100 \
  --noise_ratio 0.2 \
  --noise_type symmetric \
  --balanced \
  --num_epochs 50 \
  --adapt_epochs 5 \
  --adapt_iter_num 3 \
  --learning_rate 0.05 \
  --optimizer adam \
  --batch_size 128 \
  --eval_map
```

## Retrieval mAP and Corruption Checks

Evaluate any saved checkpoint with the same accuracy/mAP code, including Raw, Rehearsal, CONTRA, LNL baselines, and TTA baselines:

```bash
PYTHONPATH=. python scripts/evaluate_checkpoint.py \
  --step 4 \
  --model cifar-wideresnet40 \
  --dataset cifar-100 \
  --noise_ratio 0.2 \
  --noise_type asymmetric \
  --balanced \
  --model_suffix worker_tta \
  --eval_map
```

Generate corrupted request streams from the clean `test_data.npy`:

```bash
PYTHONPATH=. python gen_dataset/gen_corruption_data.py \
  --dataset pet-37 \
  --noise_ratio 0.2 \
  --noise_type symmetric \
  --balanced \
  --corruptions gaussian_noise gaussian_blur jpeg contrast
```

Then evaluate corrupted-to-clean ratios:

```bash
PYTHONPATH=. python scripts/evaluate_corruptions.py \
  --step 4 \
  --model wideresnet50 \
  --dataset pet-37 \
  --noise_ratio 0.2 \
  --noise_type symmetric \
  --balanced \
  --model_suffix worker_tta \
  --eval_map
```

## Batch Reproducibility

Print or execute three-seed runs:

```bash
PYTHONPATH=. python scripts/run_multi_seed.py \
  --method contra \
  --seeds 42 43 44 \
  --steps 0 1 2 3 4 \
  --dataset pet-37 \
  --model wideresnet50 \
  --noise_ratio 0.2 \
  --noise_type symmetric \
  --balanced \
  --eval_map
```

Print or execute hyper-parameter sweeps:

```bash
PYTHONPATH=. python scripts/run_sweep.py \
  --preset mixup_alpha \
  --values 0.2 0.4 0.8 \
  --dataset pet-37 \
  --model wideresnet50 \
  --step 4 \
  --noise_ratio 0.2 \
  --noise_type symmetric \
  --balanced \
  --eval_map
```

Available sweep presets are `mixup_alpha`, `centroid_ratio`, `conf_ratio`, `adapt_iter_num`, `rehearsal_ratio`, and `noise_ratio`. The `centroid_ratio` preset varies only the train-time nearest-centroid subset used for $\bar{D}_a^\tau$; use `conf_ratio` to vary the TTA reference subset separately. Add `--execute` to either batch script to run the printed commands.

## Ablation Commands

Print the commands used for component ablations such as w/o SN, SN on teacher+student, w/o agreement, w/o centroid guidance, and w/o Mixup:

```bash
PYTHONPATH=. python scripts/run_ablation.py \
  --dataset pet-37 \
  --model wideresnet50 \
  --step 3 \
  --noise_ratio 0.2 \
  --noise_type symmetric \
  --balanced \
  --eval_map
```

The script uses `--no_spnorm`, `--student_spnorm`, `--disable_agreement`, `--disable_centroid`, and `--disable_mixup` to map Table S7-style rows to explicit commands. Use `--execute` after reviewing the printed commands.

## License

This repository is released under the MIT license.
