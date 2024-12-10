import torch
import numpy as np
import os
import argparse

import torchvision.models
import torchvision
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


def load_pet37_superclass_mapping(file_path):
    """Load the category mapping for Oxford-Pets from a JSON file"""
    with open(file_path, "r") as f:
        pet37_superclass_to_child = json.load(f)
    return pet37_superclass_to_child


def build_asymmetric_mapping(superclass_mapping, classes, rng):
    """Build an asymmetric label mapping to ensure labels are replaced with other classes within the same superclass"""
    child_to_superclass_mapping = {}

    # Build the reverse mapping from child class to superclass
    for superclass, child_classes in superclass_mapping.items():
        for child_class in child_classes:
            child_to_superclass_mapping[child_class] = (superclass, child_classes)

    # Build the asymmetric mapping
    asymmetric_mapping = {}
    for class_name in classes:
        # Get the superclass and all categories within that superclass for the given category
        if class_name in child_to_superclass_mapping:
            superclass, child_classes = child_to_superclass_mapping[class_name]
            # Randomly select a different category within the same superclass as the replacement
            available_classes = [c for c in child_classes if c != class_name]
            if available_classes:
                new_class = rng.choice(available_classes)
                asymmetric_mapping[class_name] = new_class
            else:
                asymmetric_mapping[class_name] = (
                    class_name  # If there are no other categories, keep the original label unchanged
                )
    return asymmetric_mapping


def create_pet37_npy_files(
    data_dir,
    gen_dir,
    noise_type="symmetric",
    noise_ratio=0.2,
    num_versions=3,
    retention_ratios=[0.5, 0.3, 0.1],
    balanced=False,
):

    rng = np.random.default_rng(42)

    weights = torchvision.models.ResNet18_Weights.DEFAULT

    data_transform = transforms.Compose([weights.transforms()])

    train_dataset = datasets.OxfordIIITPet(
        root=data_dir, download=True, transform=data_transform
    )
    test_dataset = datasets.OxfordIIITPet(
        root=data_dir, split="test", download=True, transform=data_transform
    )

    train_data, train_labels = zip(*train_dataset)
    train_data = torch.stack(train_data)
    train_labels = torch.tensor(train_labels)

    test_data, test_labels = zip(*test_dataset)
    test_data = torch.stack(test_data)
    test_labels = torch.tensor(test_labels)

    case = settings.get_case(noise_ratio, noise_type, balanced)
    print("Using class-balanced data splitting...")
    dataset_name = "pet-37"
    num_classes = 37
    D_inc_data, D_inc_labels = split(
        dataset_name, case, train_dataset, test_dataset, num_classes
    )

    # Read classes of Oxford-Pets dataset
    pet37_classes_file = os.path.join(
        settings.root_dir, "configs/classes/pet_37_classes.txt"
    )
    pet37_classes = load_classes_from_file(pet37_classes_file)

    print("PET-37 Classes:", pet37_classes)

    # Read the superclass and child class mapping for Oxford-Pets
    pet37_mapping_file = os.path.join(
        settings.root_dir, "configs/classes/pet_37_mapping.json"
    )
    pet37_superclass_mapping = load_pet37_superclass_mapping(pet37_mapping_file)

    # Define the forgetting categories and noisy categories
    forget_classes = list(
        range(18)
    )  # The first 18 classes are defined as forgotten classes
    noise_classes = list(
        range(18, 28)
    )  # The next 10 classes are defined as noisy classes

    # Get the indices of forgotten and noisy classes in the incremental dataset
    D_inc_forget_indices = [
        i for i in range(len(D_inc_labels)) if D_inc_labels[i] in forget_classes
    ]
    D_inc_noise_indices = [
        i for i in range(len(D_inc_labels)) if D_inc_labels[i] in noise_classes
    ]

    if noise_type == "asymmetric":
        asymmetric_mapping = build_asymmetric_mapping(
            pet37_superclass_mapping, pet37_classes, rng
        )

    symmetric_noisy_classes = []
    asymmetric_noisy_classes = []
    symmetric_noisy_classes_simple = set()
    asymmetric_noisy_classes_simple = set()

    rng = np.random.default_rng(42)
    for t in range(num_versions):
        retention_ratio = retention_ratios[t]
        num_forget_samples = int(len(D_inc_forget_indices) * retention_ratio)
        if num_forget_samples > 0:
            forget_sample_indices = rng.choice(
                D_inc_forget_indices, num_forget_samples, replace=False
            )
            D_f_data = D_inc_data[forget_sample_indices]
            D_f_labels = D_inc_labels[forget_sample_indices]
        else:
            D_f_data = torch.empty(0, 3, 224, 224)
            D_f_labels = torch.empty(0, dtype=torch.long)

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
                if original_label in noise_classes:
                    original_class_name = pet37_classes[original_label]
                    if noise_type == "symmetric":
                        new_label = rng.choice(
                            [i for i in range(num_classes) if i != original_label]
                        )
                        D_n_labels[idx_in_D_n] = new_label
                        symmetric_noisy_classes.append(
                            {
                                "original_label": int(original_label),
                                "original_class_name": original_class_name,
                                "new_label": int(new_label),
                                "new_class_name": pet37_classes[new_label],
                            }
                        )
                        symmetric_noisy_classes_simple.add(
                            (int(original_label), int(new_label))
                        )
                    elif noise_type == "asymmetric":
                        if original_class_name in asymmetric_mapping:
                            new_class_name = asymmetric_mapping[original_class_name]
                            new_label = pet37_classes.index(new_class_name)
                            D_n_labels[idx_in_D_n] = new_label
                            asymmetric_noisy_classes.append(
                                {
                                    "original_label": int(original_label),
                                    "original_class_name": original_class_name,
                                    "new_label": int(new_label),
                                    "new_class_name": new_class_name,
                                }
                            )
                            asymmetric_noisy_classes_simple.add(
                                (int(original_label), int(new_label))
                            )
                    else:
                        raise ValueError("Invalid noise type.")

        D_tr_data = np.concatenate([D_f_data, D_n_data], axis=0)
        D_tr_labels = np.concatenate([D_f_labels, D_n_labels], axis=0)
        perm = rng.permutation(len(D_tr_data))
        D_tr_data = D_tr_data[perm]
        D_tr_labels = D_tr_labels[perm]

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

    if noise_type == "symmetric":
        with open(
            f"{dataset_name}-{noise_type}-{noise_ratio}-symmetric_noisy_classes_detailed.json",
            "w",
        ) as f:
            json.dump(symmetric_noisy_classes, f, indent=4)
        with open(
            f"{dataset_name}-{noise_type}-{noise_ratio}-symmetric_noisy_classes_simple.json",
            "w",
        ) as f:
            json.dump(list(symmetric_noisy_classes_simple), f, indent=4)
    elif noise_type == "asymmetric":
        with open(
            f"{dataset_name}-{noise_type}-{noise_ratio}-asymmetric_noisy_classes_detailed.json",
            "w",
        ) as f:
            json.dump(asymmetric_noisy_classes, f, indent=4)
        with open(
            f"{dataset_name}-{noise_type}-{noise_ratio}-asymmetric_noisy_classes_simple.json",
            "w",
        ) as f:
            json.dump(list(asymmetric_noisy_classes_simple), f, indent=4)

    print("All datasets have been generated.")


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(
        description="Generate PET-37 incremental datasets."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/pet-37/normal/",
        help="Directory of the original PET-37 dataset",
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        default="./data/pet-37/gen",
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

    args = parser.parse_args()

    create_pet37_npy_files(
        data_dir=args.data_dir,
        gen_dir=args.gen_dir,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        num_versions=args.num_versions,
        retention_ratios=args.retention_ratios,
        balanced=args.balanced,
    )


if __name__ == "__main__":
    main()
