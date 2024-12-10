import torch
import numpy as np
import os
import argparse
from torchvision import datasets, transforms
import json

import collections
from configs import settings
from split_dataset import split


def load_classes_from_file(file_path):
    """# Read the list of classes from a file"""
    with open(file_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def load_cifar100_superclass_mapping(file_path):
    """# Load the superclass to child class mapping of CIFAR-100 from a JSON file"""
    with open(file_path, "r") as f:
        cifar100_superclass_to_child = json.load(f)
    return cifar100_superclass_to_child


def build_asymmetric_mapping(superclass_mapping, classes, rng):
    """# Build an asymmetric label mapping to ensure labels are replaced with other classes within the same superclass"""
    child_to_superclass_mapping = {}

    # Build the reverse mapping from child class to superclass
    for superclass, child_classes in superclass_mapping.items():
        for child_class in child_classes:
            child_to_superclass_mapping[child_class] = (superclass, child_classes)

    # Build the asymmetric mapping table
    asymmetric_mapping = {}

    for class_name in classes:
        # Get the superclass and all classes within that superclass for the given class
        if class_name in child_to_superclass_mapping:
            superclass, child_classes = child_to_superclass_mapping[class_name]
            # Randomly select a different class within the same superclass as the replacement
            available_classes = [c for c in child_classes if c != class_name]
            if available_classes:
                new_class = rng.choice(available_classes)
                asymmetric_mapping[class_name] = new_class
            else:
                asymmetric_mapping[class_name] = (
                    class_name  # If there are no other classes, keep the original label unchanged
                )
    return asymmetric_mapping


def create_cifar100_npy_files(
    data_dir,
    gen_dir,
    noise_type="asymmetric",
    noise_ratio=0.2,
    num_versions=3,
    retention_ratios=[0.5, 0.3, 0.1],
    balanced=False,
    split_ratio=0.5,
):

    rng = np.random.default_rng(42)

    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )

    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=data_transform
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=data_transform
    )

    case = settings.get_case(noise_ratio, noise_type, balanced)

    print("Using class-balanced data splitting...")
    dataset_name = "cifar-100"
    num_classes = 100
    D_inc_data, D_inc_labels = split(
        dataset_name, case, train_dataset, test_dataset, num_classes
    )

    # Read CIFAR-100 classes
    cifar100_classes_file = os.path.join(
        settings.root_dir, "configs/classes/cifar_100_classes.txt"
    )
    cifar100_classes = load_classes_from_file(cifar100_classes_file)

    # Read the superclass to child class mapping of CIFAR-100
    cifar100_mapping_file = os.path.join(
        settings.root_dir, "configs/classes/cifar_100_mapping.json"
    )
    cifar100_superclass_mapping = load_cifar100_superclass_mapping(
        cifar100_mapping_file
    )

    print("CIFAR-100 Classes:", cifar100_classes)

    # Build the asymmetric mapping if asymmetric noise is selected
    if noise_type == "asymmetric":
        asymmetric_mapping = build_asymmetric_mapping(
            cifar100_superclass_mapping, cifar100_classes, rng
        )

    # Define the forgotten classes and noisy classes
    forget_classes = list(
        range(50)
    )  # The first 50 classes are defined as forgotten classes
    noise_classes = list(
        range(50, 75)
    )  # The next 25 classes are defined as noisy classes

    # Get the indices of forgotten and noisy classes in the incremental dataset
    D_inc_forget_indices = [
        i for i in range(len(D_inc_labels)) if D_inc_labels[i] in forget_classes
    ]
    D_inc_noise_indices = [
        i for i in range(len(D_inc_labels)) if D_inc_labels[i] in noise_classes
    ]

    symmetric_noisy_classes = []
    asymmetric_noisy_classes = []

    symmetric_noisy_classes_simple = set()
    asymmetric_noisy_classes_simple = set()

    rng = np.random.default_rng(42)

    # Generate the incremental version of the dataset
    for t in range(num_versions):
        retention_ratio = retention_ratios[t]

        # Simulate forgetting: sample from forgotten classes according to the retention ratio
        num_forget_samples = int(len(D_inc_forget_indices) * retention_ratio)

        if num_forget_samples > 0:
            forget_sample_indices = rng.choice(
                D_inc_forget_indices, num_forget_samples, replace=False
            )
            D_f_data = D_inc_data[forget_sample_indices]
            D_f_labels = D_inc_labels[forget_sample_indices]
        else:
            D_f_data = torch.empty(0, 3, 32, 32)
            D_f_labels = torch.empty(0, dtype=torch.long)

        # Noise injection: inject noise into samples of noisy classes
        noise_sample_indices = D_inc_noise_indices
        num_noisy_samples = int(len(noise_sample_indices) * noise_ratio)

        if num_noisy_samples > 0:
            noisy_indices = rng.choice(
                noise_sample_indices, num_noisy_samples, replace=False
            )
        else:
            noisy_indices = []

        D_n_data = D_inc_data[noise_sample_indices]
        D_n_labels = D_inc_labels[noise_sample_indices]

        for idx_in_D_n, D_inc_idx in enumerate(noise_sample_indices):
            if D_inc_idx in noisy_indices:
                original_label = int(D_n_labels[idx_in_D_n].item())

                # Ensure replacements are only made for classes in noise_classes
                if original_label in noise_classes:
                    original_class_name = cifar100_classes[original_label]
                    original_superclass = next(
                        (
                            superclass
                            for superclass, children in cifar100_superclass_mapping.items()
                            if original_class_name in children
                        ),
                        None,
                    )

                if noise_type == "symmetric":
                    new_label = rng.choice(
                        [i for i in range(100) if i != original_label]
                    )
                    D_n_labels[idx_in_D_n] = new_label
                    new_class_name = cifar100_classes[new_label]
                    new_superclass = next(
                        (
                            superclass
                            for superclass, children in cifar100_superclass_mapping.items()
                            if new_class_name in children
                        ),
                        None,
                    )
                    symmetric_noisy_classes.append(
                        {
                            "original_label": int(original_label),
                            "original_class_name": original_class_name,
                            "original_superclass": original_superclass,
                            "new_label": int(new_label),
                            "new_class_name": new_class_name,
                            "new_superclass": new_superclass,
                        }
                    )
                    symmetric_noisy_classes_simple.add(
                        (int(original_label), int(new_label))
                    )

                elif noise_type == "asymmetric":
                    if original_class_name in asymmetric_mapping:
                        new_class_name = asymmetric_mapping[original_class_name]
                        new_label = cifar100_classes.index(new_class_name)
                        D_n_labels[idx_in_D_n] = new_label
                        new_superclass = next(
                            (
                                superclass
                                for superclass, children in cifar100_superclass_mapping.items()
                                if new_class_name in children
                            ),
                            None,
                        )
                        asymmetric_noisy_classes.append(
                            {
                                "original_label": int(original_label),
                                "original_class_name": original_class_name,
                                "original_superclass": original_superclass,
                                "new_label": int(new_label),
                                "new_class_name": new_class_name,
                                "new_superclass": new_superclass,
                            }
                        )
                        asymmetric_noisy_classes_simple.add(
                            (int(original_label), int(new_label))
                        )
                else:
                    raise ValueError("Invalid noise type.")

        D_tr_data = np.concatenate([D_f_data, D_n_data], axis=0)
        D_tr_labels = np.concatenate([D_f_labels, D_n_labels], axis=0)

        # Shuffle the training dataset
        perm = rng.permutation(len(D_tr_data))
        D_tr_data = D_tr_data[perm]
        D_tr_labels = D_tr_labels[perm]

        # Save the training dataset
        train_data_path = settings.get_dataset_path(
            dataset_name, case, "train_data", t + 1
        )
        train_label_path = settings.get_dataset_path(
            dataset_name, case, "train_label", t + 1
        )

        subdir = os.path.dirname(train_data_path)
        os.makedirs(subdir, exist_ok=True)

        np.save(train_data_path, D_tr_data)
        np.save(train_label_path, D_tr_labels)

        print(f"Version {t+1} of D_tr has been saved to {subdir}")

    # Log the noise injection information
    if noise_type == "symmetric":
        with open(
            f"{dataset_name}_{noise_type}_{noise_ratio}_symmetric_noisy_classes_detailed.json",
            "w",
        ) as f:
            json.dump(symmetric_noisy_classes, f, indent=4)
        with open(
            f"{dataset_name}_{noise_type}_{noise_ratio}_symmetric_noisy_classes_simple.json",
            "w",
        ) as f:
            json.dump(list(symmetric_noisy_classes_simple), f, indent=4)
    elif noise_type == "asymmetric":
        with open(
            f"{dataset_name}_{noise_type}_{noise_ratio}_asymmetric_noisy_classes_detailed.json",
            "w",
        ) as f:
            json.dump(asymmetric_noisy_classes, f, indent=4)
        with open(
            f"{dataset_name}_{noise_type}_{noise_ratio}_asymmetric_noisy_classes_simple.json",
            "w",
        ) as f:
            json.dump(list(asymmetric_noisy_classes_simple), f, indent=4)

    print("All datasets have been generated.")


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(
        description="Generate CIFAR-100 incremental datasets."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/cifar-100/normal",
        help="Directory of the original CIFAR-100 dataset",
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        default="./data/cifar-100/gen/",
        help="Directory to save the generated datasets",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        choices=["symmetric", "asymmetric"],
        default="symmetric",
        help="Label noise type: 'symmetric' or 'asymmetric'",
    )
    parser.add_argument(
        "--noise_ratio",
        type=float,
        default=0.2,
        help="Noise ratio (default 0.2)",
    )
    parser.add_argument(
        "--num_versions",
        type=int,
        default=3,
        help="Number of incremental versions to generate (default 3)",
    )
    parser.add_argument(
        "--retention_ratios",
        type=float,
        nargs="+",
        default=[0.5, 0.3, 0.1],
        help="List of retention ratios for each incremental version",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Whether to use class-balanced data splitting. If not specified, random splitting is used.",
    )

    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.5,
        help="Ratio for class-balanced splitting, default is 0.5",
    )

    args = parser.parse_args()

    create_cifar100_npy_files(
        data_dir=args.data_dir,
        gen_dir=args.gen_dir,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        num_versions=args.num_versions,
        retention_ratios=args.retention_ratios,
        balanced=args.balanced,
        split_ratio=args.split_ratio,
    )


if __name__ == "__main__":
    main()
