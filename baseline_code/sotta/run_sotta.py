"""Entry script for SoTTA test-time adaptation.

Usage (example):
    python baseline_code/sotta/run_sotta.py \
        --dataset cifar-10 --model resnet18 --batch_size 200 \
        --noise_ratio 0.2 --noise_type sym --balanced 1 \
        --step 1 --uni_name SoTTA
"""

import logging
import os
import shutil
import sys

import numpy as np
import torch
import torch.optim as optim

from robustbench.utils import clean_accuracy as accuracy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import sotta
from core_model.dataset import get_dataset_loader
from args_paser import parse_args
from configs import settings
from core_model.custom_model import load_custom_model, ClassifierWrapper

logger = logging.getLogger(__name__)


def main():
    custom_args = parse_args()
    case = settings.get_case(
        custom_args.noise_ratio, custom_args.noise_type, custom_args.balanced)
    step = getattr(custom_args, "step", 1)
    uni_name = getattr(custom_args, "uni_name", "SoTTA")
    num_classes = settings.num_classes_dict[custom_args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data, test_labels, testloader = get_dataset_loader(
        custom_args.dataset, "test", case, None, None, None,
        custom_args.batch_size, shuffle=False)

    load_model_path = settings.get_ckpt_path(
        custom_args.dataset, case, custom_args.model,
        model_suffix="worker_restore", step=0, unique_name=uni_name)

    if not os.path.exists(load_model_path):
        contra_model_path = settings.get_ckpt_path(
            custom_args.dataset, case, custom_args.model,
            model_suffix="worker_restore", step=0, unique_name="contra")
        os.makedirs(os.path.dirname(load_model_path), exist_ok=True)
        shutil.copy(contra_model_path, load_model_path)
        print("copy contra model: %s to : %s" % (
            contra_model_path, load_model_path))

    save_model_path = settings.get_ckpt_path(
        custom_args.dataset, case, custom_args.model,
        model_suffix="worker_tta", step=step, unique_name=uni_name)

    loaded_model = load_custom_model(
        custom_args.model, num_classes, load_pretrained=False)
    base_model = ClassifierWrapper(loaded_model, num_classes)
    checkpoint = torch.load(load_model_path)
    base_model.load_state_dict(checkpoint, strict=False)
    base_model.to(device)

    base_model = sotta.configure_model(base_model)
    params, param_names = sotta.collect_params(base_model)
    optimizer = optim.Adam(params, lr=1e-3, weight_decay=0.0)

    model = sotta.SoTTA(
        base_model, optimizer,
        steps=1,
        episodic=False,
        conf_threshold=0.9,
        num_classes=num_classes,
        bank_size=64,
        rho=0.05,
    )
    logger.info("SoTTA model configured, params: %s", param_names)

    try:
        model.reset()
        logger.info("resetting model")
    except Exception:
        logger.warning("not resetting model")

    x_test = torch.from_numpy(test_data)
    y_test = torch.from_numpy(test_labels)
    x_test, y_test = x_test.to(device), y_test.to(device)

    acc = accuracy(model, x_test, y_test, custom_args.batch_size,
                   save_path=save_model_path)
    err = 1.0 - acc
    logger.info("SoTTA accuracy: %.4f, error: %.2f%%", acc, err * 100)
    print("SoTTA accuracy: %.4f, error: %.2f%%" % (acc, err * 100))


if __name__ == "__main__":
    main()
