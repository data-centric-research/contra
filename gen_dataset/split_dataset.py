import os
import shutil

import numpy as np
import torch

from configs import settings


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def split_by_class(data, labels, num_classes):
    """Group samples by class label."""
    class_data = {i: [] for i in range(num_classes)}
    for index, label in enumerate(labels):
        class_data[int(label)].append(data[index])
    return class_data


def sample_class_balanced_data(class_data, split_ratio):
    """Split each class into an initial clean set and an incremental pool."""
    initial_data, initial_labels = [], []
    incremental_data, incremental_labels = [], []

    for class_label, samples in class_data.items():
        num_samples = len(samples)
        split_index = int(num_samples * split_ratio)
        shuffled_indices = np.random.permutation(num_samples)

        initial_data.extend(samples[i] for i in shuffled_indices[:split_index])
        initial_labels.extend([class_label] * split_index)

        incremental_data.extend(samples[i] for i in shuffled_indices[split_index:])
        incremental_labels.extend([class_label] * (num_samples - split_index))

    return (
        np.stack([_to_numpy(item) for item in initial_data]),
        np.asarray(initial_labels, dtype=np.int64),
        np.stack([_to_numpy(item) for item in incremental_data]),
        np.asarray(incremental_labels, dtype=np.int64),
    )


def sample_rehearsal_data(initial_data, initial_labels, rehearsal_ratio, num_classes):
    """Sample the fixed rehearsal buffer from the clean initial set."""
    class_data = split_by_class(initial_data, initial_labels, num_classes)
    rehearsal_data, rehearsal_labels = [], []

    for class_label, samples in class_data.items():
        num_samples = len(samples)
        num_rehearsal_samples = int(num_samples * rehearsal_ratio)
        if num_rehearsal_samples == 0:
            continue

        sample_indices = np.random.choice(
            num_samples, num_rehearsal_samples, replace=False
        )
        rehearsal_data.extend(samples[i] for i in sample_indices)
        rehearsal_labels.extend([class_label] * num_rehearsal_samples)

    return (
        np.stack([_to_numpy(item) for item in rehearsal_data]),
        np.asarray(rehearsal_labels, dtype=np.int64),
    )


def _load_torchvision_dataset(dataset):
    data, labels = zip(*dataset)
    return torch.stack(data), torch.tensor(labels)


def split(
    dataset_name,
    case,
    train_dataset=None,
    test_dataset=None,
    num_classes=100,
    split_ratio=0.5,
    rehearsal_ratio=0.1,
):
    """Create paper-aligned initial, incremental, rehearsal, and test files."""
    raw_case = None
    train_data_path = settings.get_dataset_path(dataset_name, raw_case, "train_data")
    train_label_path = settings.get_dataset_path(dataset_name, raw_case, "train_label")
    test_data_path = settings.get_dataset_path(dataset_name, raw_case, "test_data")
    test_label_path = settings.get_dataset_path(dataset_name, raw_case, "test_label")
    aux_data_path = settings.get_dataset_path(dataset_name, raw_case, "aux_data")
    aux_label_path = settings.get_dataset_path(dataset_name, raw_case, "aux_label")
    inc_data_path = settings.get_dataset_path(dataset_name, raw_case, "inc_data")
    inc_label_path = settings.get_dataset_path(dataset_name, raw_case, "inc_label")
    initial_data_path = settings.get_dataset_path(
        dataset_name, raw_case, "train_0_data"
    )
    initial_label_path = settings.get_dataset_path(
        dataset_name, raw_case, "train_0_label"
    )

    needs_split = not os.path.exists(initial_label_path)
    if needs_split:
        if train_dataset is None or test_dataset is None:
            raise FileNotFoundError(
                "Generated split files are missing and no torchvision dataset was provided."
            )

        train_data, train_labels = _load_torchvision_dataset(train_dataset)
        test_data, test_labels = _load_torchvision_dataset(test_dataset)

        class_data = split_by_class(train_data, train_labels, num_classes)
        initial_data, initial_labels, inc_data, inc_labels = sample_class_balanced_data(
            class_data, split_ratio=split_ratio
        )
        aux_data, aux_labels = sample_rehearsal_data(
            initial_data,
            initial_labels,
            rehearsal_ratio=rehearsal_ratio,
            num_classes=num_classes,
        )

        os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
        np.save(train_data_path, _to_numpy(train_data))
        np.save(train_label_path, _to_numpy(train_labels).astype(np.int64))
        np.save(test_data_path, _to_numpy(test_data))
        np.save(test_label_path, _to_numpy(test_labels).astype(np.int64))
        np.save(aux_data_path, aux_data)
        np.save(aux_label_path, aux_labels)
        np.save(inc_data_path, inc_data)
        np.save(inc_label_path, inc_labels)
        np.save(initial_data_path, initial_data)
        np.save(initial_label_path, initial_labels)

    case_test_data_path = settings.get_dataset_path(dataset_name, case, "test_data")
    case_test_label_path = settings.get_dataset_path(dataset_name, case, "test_label")
    case_aux_data_path = settings.get_dataset_path(dataset_name, case, "aux_data")
    case_aux_label_path = settings.get_dataset_path(dataset_name, case, "aux_label")
    case_initial_data_path = settings.get_dataset_path(
        dataset_name, case, "train_data", step=0
    )
    case_initial_label_path = settings.get_dataset_path(
        dataset_name, case, "train_label", step=0
    )

    os.makedirs(os.path.dirname(case_initial_data_path), exist_ok=True)
    shutil.copy(test_data_path, case_test_data_path)
    shutil.copy(test_label_path, case_test_label_path)
    shutil.copy(aux_data_path, case_aux_data_path)
    shutil.copy(aux_label_path, case_aux_label_path)
    shutil.copy(initial_data_path, case_initial_data_path)
    shutil.copy(initial_label_path, case_initial_label_path)

    return np.load(inc_data_path), np.load(inc_label_path)
