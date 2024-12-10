
# CONTRA: A Continual Restoration and Adaptation Learning Framework for Ever-robust Online Image Recognition and Search

<div align="center">

 [:sparkles: Overview](#sparkles-overview) | [:computer: Usage](#computer-usage) | [:link: License](#link-License)

<div align="left">

<!-- ADD WARNING: Only the core code is retained. If needed, we can provide Jupyter notebooks for baseline reproduction and result analysis, as well as pre-trained models. -->
## :sparkles: Overview

<!-- insert the pdf of framework from assets/framework.pdf -->
<img src="assets/framework.png" width="100%" style="border: none;"></img>

This repository provides a framework for noise-robust training and incremental learning with datasets such as CIFAR-10, CIFAR-100, and Oxford-IIIT Pet (PET-37). The project demonstrates robust model training under various noisy label conditions and supports multiple training modes, including raw training and incremental training.

## :computer: Usage

### :rainbow: Create Environment

Create and activate a virtual environment with the required dependencies:

```bash
conda create -n ${env_name} -r requirements.txt
conda activate ${env_name}
```

### :rocket: Prepare Dataset

#### :zap: Supported Datasets

This project supports the following datasets:

1. **CIFAR-10**: 
   - A dataset containing 60,000 32x32 color images in 10 classes, with 6,000 images per class. 
   - Directory structure:
     - `data/cifar-10/normal/`: Contains the original dataset.
     - `data/cifar-10/gen/`: Stores generated datasets with noise.

2. **CIFAR-100**:
   - A dataset containing 60,000 32x32 color images in 100 classes, grouped into 20 superclasses.
   - Directory structure:
     - `data/cifar-100/normal/`: Contains the original dataset.
     - `data/cifar-100/gen/`: Stores generated datasets with noise.

3. **Oxford-IIIT Pet (PET-37)**:
   - A dataset containing 37 categories of pet images with approximately 200 images per category.
   - Directory structure:
     - `data/pet-37/normal/oxford-pets/`: Contains the original dataset.
     - `data/pet-37/gen/`: Stores generated datasets with noise.

#### :zap: Generate Noisy Dataset

You can generate datasets with symmetric or asymmetric label noise for any supported dataset by running the provided scripts.

**Example command for CIFAR-100 (symmetric noise):**

```bash
PYTHONPATH=${code_base} python gen_dataset/gen_cifar100_exp_data.py \
--data_dir ./data/cifar-100/normal \
--gen_dir ./data/cifar-100/gen/ \
--noise_type symmetric \
--noise_ratio 0.2 \
--num_versions 3 \
--retention_ratios 0.5 0.3 0.1 \
--balanced
```

**Replace `--data_dir` and `--gen_dir` with paths for CIFAR-10 or PET-37 as needed.**

### :fire: Training Models

#### :zap: Training Modes

The framework supports the following training modes across all datasets:

1. **Raw Training**: Train models on datasets with no incremental steps.

**Example command for CIFAR-100:**

```bash
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_NUM} python run_experiment.py \
--step -1 \
--model cifar-wideresnet40 \
--dataset cifar-100 \
--noise_ratio 0.2 \
--noise_type symmetric \
--balanced \
--epoch 200 \
--learning_rate 0.05 \
--optimizer adam \
--batch_size 256
```

2. **Incremental Training**: Train models incrementally on noisy datasets.

**Example commands for CIFAR-10 (symmetric noise):**

- Step 0: Train M_p0

```bash
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_NUM} python run_experiment.py \
--step 0 \
--model cifar-wideresnet40 \
--dataset cifar-10 \
--noise_ratio 0.2 \
--noise_type symmetric \
--balanced \
--epoch 50 \
--learning_rate 0.05 \
--optimizer adam \
--batch_size 256
```

- Step 1: Train M_p1

```bash
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_NUM} python run_experiment.py \
--step 1 \
--model cifar-wideresnet40 \
--dataset cifar-10 \
--noise_ratio 0.2 \
--noise_type symmetric \
--balanced \
--epoch 50 \
--learning_rate 0.05 \
--optimizer adam \
--batch_size 256 \
--load_model_path ckpt/cifar-10/nr_0.2_nt_symmetric/model_p0.pth
```

- Repeat for Steps 2 and 3 using the appropriate dataset and retention ratios.

**Replace dataset-specific arguments with those for CIFAR-100 or PET-37 as needed.**

### :bar_chart: Evaluation

After training, you can evaluate results by checking the generated data and trained models.

```bash
# Check generated datasets
tree data/cifar-10/gen/
tree data/cifar-100/gen/
tree data/pet-37/gen/

# Check trained model checkpoints
tree ckpt/cifar-10/nr_0.2_nt_symmetric/
tree ckpt/cifar-100/nr_0.2_nt_symmetric/
tree ckpt/pet-37/nr_0.2_nt_symmetric/
```

### :scroll: License

This repository is released under the MIT license.
