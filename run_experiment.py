import os
import shutil
import warnings
import numpy as np
from args_paser import parse_args

from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from core_model.optimizer import create_optimizer_scheduler
from core_model.custom_model import ClassifierWrapper, load_custom_model
from configs import settings

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms import v2
from train_test_utils import train_model


def get_num_of_classes(dataset_name):
    if dataset_name == "cifar-10":
        num_classes = 10
    elif dataset_name == "pet-37":
        num_classes = 37
    elif dataset_name == "cifar-100":
        num_classes = 100
    elif dataset_name == "food-101":
        num_classes = 101
    elif dataset_name == "flower-102":
        num_classes = 102
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_name}")

    return num_classes


def load_dataset(file_path, is_data=True):
    """
    Load dataset file and return it as a PyTorch tensor.
    :param file_path: Path to the dataset file
    :param is_data: Boolean indicating whether the file is a data file (True for data file, False for label file)
    :return: Data in PyTorch tensor format
    """
    data = np.load(file_path)

    if is_data:
        # For data files, convert to float32 type
        data_tensor = torch.tensor(data, dtype=torch.float32)
    else:
        # For label files, convert to long type
        data_tensor = torch.tensor(data, dtype=torch.long)

    return data_tensor


def train_step(
    args,
    writer=None,
):
    """
    Train the model according to the specified step.
    :param step: The step to execute (0, 1, 2, ...)
    :param subdir: Path to the data subdirectory
    :param ckpt_subdir: Path to the model checkpoint subdirectory
    :param output_dir: Directory to save the model
    :param dataset_name: Type of dataset to use (cifar-10 or cifar-100)
    :param load_model_path: Path to the model to load (optional)
    :param epochs: Number of training epochs
    :param batch_size: Batch size
    :param optimizer_type: Optimizer
    :param learning_rate: Learning rate
    """
    warnings.filterwarnings("ignore")

    # num_classes = 10 if dataset_name == "cifar-10" else 100
    dataset_name = args.dataset
    num_classes = get_num_of_classes(dataset_name)

    print(f"===== Executing step: {args.step} =====")
    print(f"Dataset: {dataset_name}")
    print(
        f"Epochs: {args.num_epochs}, Batch Size: {args.batch_size}, Learning Rate: {args.learning_rate}"
    )

    model_name = args.model
    step = args.step
    case = settings.get_case(args.noise_ratio, args.noise_type, args.balanced)
    uni_name = args.uni_name
    model_suffix = "worker_raw" if args.model_suffix is None else args.model_suffix
    if step < 0:

        D_train_data = load_dataset(
            settings.get_dataset_path(dataset_name, case, "train_data")
        )
        D_train_labels = np.load(
            settings.get_dataset_path(dataset_name, case, "train_label")
        )
        D_test_data = np.load(
            settings.get_dataset_path(dataset_name, case, "test_data")
        )
        D_test_labels = np.load(
            settings.get_dataset_path(dataset_name, case, "test_label")
        )

        # Print the model and data used for training
        print("Data used for training: train_data.npy and train_labels.npy")
        print("Model used for training: ResNet18 initialized")

        model_raw = load_custom_model(model_name, num_classes)
        model_raw = ClassifierWrapper(
            model_raw, num_classes=num_classes, freeze_weights=False
        )
        print(f"Start training model M_raw on ({dataset_name})...")

        model_raw = train_model(
            model_raw,
            num_classes,
            D_train_data,
            D_train_labels,
            D_test_data,
            D_test_labels,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            data_aug=args.data_aug,
            writer=writer,
        )
        model_raw_path = settings.get_ckpt_path(
            dataset_name, case, model_name, "worker_restore", unique_name=uni_name
        )
        subdir = os.path.dirname(model_raw_path)
        os.makedirs(subdir, exist_ok=True)
        torch.save(model_raw.state_dict(), model_raw_path)
        print(f"M_raw has been trained and save to {model_raw_path}")
        return
    elif (
        step == 0
    ):  # Train a model M_p0 based on the $D_0$ dataset and the original ResNet network
        pretrain_case = "pretrain"
        model_p0_path = settings.get_ckpt_path(
            dataset_name, pretrain_case, model_name, "worker_restore", step=step
        )

        if uni_name is None:
            D_train_data = np.load(
                settings.get_dataset_path(dataset_name, case, "train_data", step=step)
            )
            D_train_labels = np.load(
                settings.get_dataset_path(dataset_name, case, "train_label", step=step)
            )
            D_test_data = np.load(
                settings.get_dataset_path(dataset_name, case, "test_data")
            )
            D_test_labels = np.load(
                settings.get_dataset_path(dataset_name, case, "test_label")
            )

            print("Data used for training: D_0.npy 和 D_0_labels.npy")
            print("Model used for training: ResNet18")

            model_p0 = load_custom_model(model_name, num_classes)
            model_p0 = ClassifierWrapper(model_p0, num_classes)

            print(f"Start training M_p0 on ({dataset_name})...")

            model_p0 = train_model(
                model_p0,
                num_classes,
                D_train_data,
                D_train_labels,
                D_test_data,
                D_test_labels,
                epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                data_aug=args.data_aug,
                writer=writer,
            )
            subdir = os.path.dirname(model_p0_path)
            os.makedirs(subdir, exist_ok=True)
            torch.save(model_p0.state_dict(), model_p0_path)
            print(f"M_p0 has been trained and save to {model_p0_path}")
        else:
            copy_model_p0_path = settings.get_ckpt_path(
                dataset_name,
                case,
                model_name,
                "worker_restore",
                step=step,
                unique_name=uni_name,
            )
            if os.path.exists(model_p0_path):
                subdir = os.path.dirname(copy_model_p0_path)
                os.makedirs(subdir, exist_ok=True)
                shutil.copy(model_p0_path, copy_model_p0_path)
                print(f"Copy {model_p0_path} to {copy_model_p0_path}")
            else:
                raise FileNotFoundError(model_p0_path)

    else:
        if args.train_aux:
            trainset = "aux"
            load_model_suffix = "worker_raw"
            data_step = None
            model_step = step
        else:
            trainset = "train"
            load_model_suffix = "worker_restore"
            data_step = step
            model_step = step - 1
        D_train_data = np.load(
            settings.get_dataset_path(
                dataset_name, case, f"{trainset}_data", step=data_step
            )
        )
        D_train_labels = np.load(
            settings.get_dataset_path(
                dataset_name, case, f"{trainset}_label", step=data_step
            )
        )
        D_test_data = np.load(
            settings.get_dataset_path(dataset_name, case, "test_data")
        )
        D_test_labels = np.load(
            settings.get_dataset_path(dataset_name, case, "test_label")
        )

        print(f"Model used for training: M_p{step-1}")

        prev_model_path = settings.get_ckpt_path(
            dataset_name,
            case,
            model_name,
            load_model_suffix,
            step=model_step,
            unique_name=uni_name,
        )
        print(f"Load model from: {prev_model_path}")

        if not os.path.exists(prev_model_path):
            raise FileNotFoundError(
                f"Model {prev_model_path} not found, pls train M_p{step-1} first."
            )

        model_loaded = load_custom_model(
            model_name=model_name, num_classes=num_classes, load_pretrained=False
        )
        current_model = ClassifierWrapper(model_loaded, num_classes)
        current_model.load_state_dict(torch.load(prev_model_path))

        print(f"Start training M_p{step} on ({dataset_name})...")

        current_model = train_model(
            current_model,
            num_classes,
            D_train_data,
            D_train_labels,
            D_test_data,
            D_test_labels,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            optimizer_type=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            data_aug=args.data_aug,
            writer=writer,
        )

        # save current model
        current_model_path = settings.get_ckpt_path(
            dataset_name,
            case,
            model_name,
            model_suffix,
            step=step,
            unique_name=uni_name,
        )
        subdir = os.path.dirname(current_model_path)
        os.makedirs(subdir, exist_ok=True)
        torch.save(current_model.state_dict(), current_model_path)
        print(f"M_p{step} has been trained and save to {current_model_path}")


def main():
    args = parse_args()

    writer = SummaryWriter(log_dir="runs/experiment") if args.use_tensorboard else None

    train_step(
        args,
        writer=writer,
    )

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
