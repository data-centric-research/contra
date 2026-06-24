FROM python:3.10.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade \
        pip==25.2 \
        setuptools==80.9.0 \
        wheel==0.45.1

COPY requirements-docker.txt /tmp/requirements-docker.txt

RUN python -m pip install -r /tmp/requirements-docker.txt \
    && python -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.9.1+cpu \
        torchvision==0.24.1+cpu

COPY . .

RUN python -m py_compile \
        args_paser.py \
        run_experiment.py \
        run_contra.py \
        core_model/core.py \
        core_model/train_test.py \
        gen_dataset/gen_cifar10_exp_data.py \
        gen_dataset/gen_cifar100_exp_data.py \
        gen_dataset/gen_pet37_exp_data.py \
        gen_dataset/gen_webvision_exp_data.py \
        gen_dataset/gen_corruption_data.py \
        scripts/evaluate_checkpoint.py \
        scripts/evaluate_corruptions.py \
        scripts/run_multi_seed.py \
        scripts/run_sweep.py

CMD ["bash"]
