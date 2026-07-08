import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs import settings
from gen_dataset.protocol import create_staged_tensor_protocol


def _parse_list_line(line):
    line = line.strip()
    if not line:
        return None
    if "," in line:
        path, label = [part.strip() for part in line.rsplit(",", 1)]
    else:
        path, label = line.rsplit(maxsplit=1)
    return path, label


def _read_list_file(list_path):
    samples = []
    with open(list_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parsed = _parse_list_line(line)
            if parsed is not None:
                samples.append(parsed)
    return samples


def _read_folder(root):
    class_names = [
        name
        for name in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, name))
    ]
    samples = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for class_name in class_names:
        class_dir = os.path.join(root, class_name)
        for current_root, _, files in os.walk(class_dir):
            for filename in sorted(files):
                if os.path.splitext(filename)[1].lower() in extensions:
                    samples.append(
                        (os.path.join(current_root, filename), class_name)
                    )
    return samples


def _make_label_map(train_samples, test_samples, num_classes=None):
    labels = []
    for _, label in train_samples + test_samples:
        labels.append(label)

    if all(str(label).isdigit() for label in labels):
        raw_labels = sorted({int(label) for label in labels})
        if num_classes is not None:
            raw_labels = raw_labels[:num_classes]
        return {str(label): index for index, label in enumerate(raw_labels)}

    raw_labels = sorted(set(labels))
    if num_classes is not None:
        raw_labels = raw_labels[:num_classes]
    return {label: index for index, label in enumerate(raw_labels)}


def _resolve_path(path, image_root):
    if os.path.isabs(path):
        return path
    if image_root is None:
        return path
    return os.path.join(image_root, path)


def _load_samples(samples, label_map, image_root, transform, max_per_class=None):
    per_class_counts = {label: 0 for label in label_map.values()}
    data, labels = [], []

    for image_path, raw_label in samples:
        raw_key = str(raw_label)
        if raw_key not in label_map:
            continue
        label = label_map[raw_key]
        if max_per_class is not None and per_class_counts[label] >= max_per_class:
            continue

        resolved_path = _resolve_path(image_path, image_root)
        with Image.open(resolved_path) as image:
            image = image.convert("RGB")
            data.append(transform(image).numpy())
            labels.append(label)
        per_class_counts[label] += 1

    if not data:
        raise ValueError("No images were loaded. Check paths, labels, and num_classes.")
    return np.stack(data).astype(np.float32), np.asarray(labels, dtype=np.int64)


def create_imagefolder_npy_files(
    dataset_name,
    train_root=None,
    test_root=None,
    train_list=None,
    test_list=None,
    image_root=None,
    num_classes=None,
    image_size=224,
    max_train_per_class=None,
    max_test_per_class=None,
    noise_type="symmetric",
    noise_ratio=0.2,
    num_versions=4,
    retention_ratios=None,
    split_ratio=0.5,
    rehearsal_ratio=0.1,
    balanced=False,
    seed=42,
):
    if dataset_name not in settings.dataset_paths:
        settings.dataset_paths[dataset_name] = os.path.join(
            settings.root_dir, "data", dataset_name
        )

    transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if train_list:
        train_samples = _read_list_file(train_list)
    elif train_root:
        train_samples = _read_folder(train_root)
    else:
        raise ValueError("Provide either --train_root or --train_list.")

    if test_list:
        test_samples = _read_list_file(test_list)
    elif test_root:
        test_samples = _read_folder(test_root)
    else:
        raise ValueError("Provide either --test_root or --test_list.")

    label_map = _make_label_map(train_samples, test_samples, num_classes)
    effective_num_classes = len(label_map)

    train_data, train_labels = _load_samples(
        train_samples,
        label_map,
        image_root,
        transform,
        max_per_class=max_train_per_class,
    )
    test_data, test_labels = _load_samples(
        test_samples,
        label_map,
        image_root,
        transform,
        max_per_class=max_test_per_class,
    )

    dataset_root = settings.dataset_paths.get(
        dataset_name, os.path.join(settings.root_dir, "data", dataset_name)
    )
    class_map_path = os.path.join(
        dataset_root, "gen", "class_map.json"
    )
    os.makedirs(os.path.dirname(class_map_path), exist_ok=True)
    with open(class_map_path, "w", encoding="utf-8") as handle:
        json.dump(label_map, handle, indent=2)

    case = create_staged_tensor_protocol(
        dataset_name=dataset_name,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        num_classes=effective_num_classes,
        noise_type=noise_type,
        noise_ratio=noise_ratio,
        num_versions=num_versions,
        retention_ratios=retention_ratios,
        split_ratio=split_ratio,
        rehearsal_ratio=rehearsal_ratio,
        balanced=balanced,
        seed=seed,
        metadata={"class_map_path": class_map_path, "image_size": image_size},
    )
    print(f"Generated {dataset_name} staged tensors for case: {case}")
    print(f"Effective class count: {effective_num_classes}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate staged tensor files from WebVision/Food-101/ImageFolder-style raw images."
    )
    parser.add_argument("--dataset", default="webvision")
    parser.add_argument("--train_root", default=None)
    parser.add_argument("--test_root", default=None)
    parser.add_argument("--train_list", default=None)
    parser.add_argument("--test_list", default=None)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_train_per_class", type=int, default=None)
    parser.add_argument("--max_test_per_class", type=int, default=None)
    parser.add_argument("--noise_type", choices=["symmetric", "asymmetric"], default="symmetric")
    parser.add_argument("--noise_ratio", type=float, default=0.2)
    parser.add_argument("--num_versions", type=int, default=4)
    parser.add_argument(
        "--retention_ratios",
        type=float,
        nargs="+",
        default=[0.5, 0.3, 0.1, 0.05],
    )
    parser.add_argument("--split_ratio", type=float, default=0.5)
    parser.add_argument("--rehearsal_ratio", type=float, default=0.1)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    create_imagefolder_npy_files(
        dataset_name=args.dataset,
        train_root=args.train_root,
        test_root=args.test_root,
        train_list=args.train_list,
        test_list=args.test_list,
        image_root=args.image_root,
        num_classes=args.num_classes,
        image_size=args.image_size,
        max_train_per_class=args.max_train_per_class,
        max_test_per_class=args.max_test_per_class,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        num_versions=args.num_versions,
        retention_ratios=args.retention_ratios,
        split_ratio=args.split_ratio,
        rehearsal_ratio=args.rehearsal_ratio,
        balanced=args.balanced,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
