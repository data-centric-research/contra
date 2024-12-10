import json
import torch
import numpy as np
import os
import argparse
from torchvision import datasets, transforms
from configs import settings
from split_dataset import split


def create_cifar10_npy_files(
    data_dir,
    gen_dir,
    noise_type="symmetric",
    noise_ratio=0.2,
    num_versions=3,
    retention_ratios=[0.5, 0.3, 0.1],
    balanced=False,  # choose if class balance
):

    rng = np.random.default_rng(42)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.491, 0.482, 0.446], [0.247, 0.243, 0.261]),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    case = settings.get_case(noise_ratio, noise_type, balanced)
    print("split dataset in balanced class...")
    dataset_name = "cifar-10"
    num_classes = 10
    D_inc_data, D_inc_labels = split(
        dataset_name, case, train_dataset, test_dataset, num_classes
    )

    # Define the forgetting class and noise class
    forget_classes = [1, 3, 5, 7, 9]
    noise_classes = [0, 2, 4, 6, 8]

    # Get the indices of the forgetting class and noise class in the incremental dataset
    D_inc_forget_indices = [
        i for i in range(len(D_inc_labels)) if D_inc_labels[i] in forget_classes
    ]
    D_inc_noise_indices = [
        i for i in range(len(D_inc_labels)) if D_inc_labels[i] in noise_classes
    ]

    # Define the mapping relationship of asymmetric noises
    asymmetric_mapping = {
        0: 2,
        2: 0,
        4: 6,
        6: 4,
        8: 0,
    }
    symmetric_noisy_classes = []
    asymmetric_noisy_classes = []
    symmetric_noisy_classes_simple = set()
    asymmetric_noisy_classes_simple = set()

    # generate incremental dataset
    for t in range(num_versions):
        retention_ratio = retention_ratios[t]

        # Simulate forgetting: Sample from the forgetting class according to the retention ratio
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

        # Noise injection: Inject noise into the samples of the noise class
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

        # inject noises into D_n_labels
        for idx_in_D_n, D_inc_idx in enumerate(noise_sample_indices):
            if D_inc_idx in noisy_indices:
                original_label = D_n_labels[idx_in_D_n].item()
                if noise_type == "symmetric":
                    new_label = original_label
                    while new_label == original_label:
                        new_label = rng.choice(
                            [i for i in range(num_classes) if i != original_label]
                        )
                    D_n_labels[idx_in_D_n] = new_label
                    symmetric_noisy_classes.append(
                        {
                            "original_label": int(original_label),
                            "new_label": int(new_label),
                        }
                    )
                    symmetric_noisy_classes_simple.add(
                        (int(original_label), int(new_label))
                    )
                elif noise_type == "asymmetric":
                    if original_label in asymmetric_mapping:
                        new_label = asymmetric_mapping[original_label]
                        D_n_labels[idx_in_D_n] = new_label
                        asymmetric_noisy_classes.append(
                            {
                                "original_label": int(original_label),
                                "new_label": int(new_label),
                            }
                        )
                        asymmetric_noisy_classes_simple.add(
                            (int(original_label), int(new_label))
                        )
                else:
                    raise ValueError("Invalid noise type.")

        # make D_tr^{(t)}
        D_tr_data = np.concatenate([D_f_data, D_n_data], axis=0)
        D_tr_labels = np.concatenate([D_f_labels, D_n_labels], axis=0)

        # Shuffle the training dataset
        perm = torch.randperm(len(D_tr_data))
        D_tr_data = D_tr_data[perm]
        D_tr_labels = D_tr_labels[perm]

        # save dataset
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

        print(f"D_tr {t+1} has been saved to {subdir}")

    # Save the noise injection details
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

    print("All processes of generating datasets have been saved.")


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(
        description="Generate CIFAR-10 incremental datasets."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/cifar-10/normal",
        help="Original CIFAR-10 directory",
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        default="./data/cifar-10/gen/",
        help="# Directory to save the generated dataset",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        choices=["symmetric", "asymmetric"],
        default="symmetric",
        help="Noise Type: 'symmetric' 或 'asymmetric'",
    )
    parser.add_argument(
        "--noise_ratio", type=float, default=0.2, help="Noise ratio(default 0.2)"
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

    if args.gen_dir is None:
        base_data_dir = os.path.join(os.path.dirname(__file__), "../data/cifar-10/")
        args.gen_dir = os.path.join(base_data_dir, "gen")

    create_cifar10_npy_files(
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
