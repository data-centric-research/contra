import json
import os

import numpy as np

from configs import settings


def _ensure_parent(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _save_dataset_file(dataset, case, name, value, step=None):
    path = settings.get_dataset_path(dataset, case, name, step=step)
    _ensure_parent(path)
    np.save(path, value)
    return path


def _balanced_initial_incremental_split(data, labels, num_classes, split_ratio, rng):
    initial_data, initial_labels, inc_data, inc_labels = [], [], [], []
    labels = np.asarray(labels, dtype=np.int64)

    for label in range(num_classes):
        indices = np.flatnonzero(labels == label)
        if len(indices) == 0:
            continue
        indices = rng.permutation(indices)
        split_at = int(len(indices) * split_ratio)
        if split_at <= 0 and len(indices) > 1:
            split_at = 1

        init_idx = indices[:split_at]
        inc_idx = indices[split_at:]

        initial_data.extend(data[init_idx])
        initial_labels.extend([label] * len(init_idx))
        inc_data.extend(data[inc_idx])
        inc_labels.extend([label] * len(inc_idx))

    return (
        np.stack(initial_data).astype(data.dtype, copy=False),
        np.asarray(initial_labels, dtype=np.int64),
        np.stack(inc_data).astype(data.dtype, copy=False),
        np.asarray(inc_labels, dtype=np.int64),
    )


def _sample_rehearsal(initial_data, initial_labels, num_classes, rehearsal_ratio, rng):
    rehearsal_data, rehearsal_labels = [], []
    for label in range(num_classes):
        indices = np.flatnonzero(initial_labels == label)
        if len(indices) == 0:
            continue
        sample_count = int(len(indices) * rehearsal_ratio)
        if sample_count <= 0:
            sample_count = 1
        sample_count = min(sample_count, len(indices))
        selected = rng.choice(indices, sample_count, replace=False)
        rehearsal_data.extend(initial_data[selected])
        rehearsal_labels.extend([label] * len(selected))

    return (
        np.stack(rehearsal_data).astype(initial_data.dtype, copy=False),
        np.asarray(rehearsal_labels, dtype=np.int64),
    )


def _inject_noise(labels, sample_indices, noise_type, noise_ratio, num_classes, rng):
    noisy_labels = labels.copy()
    sample_indices = np.asarray(sample_indices)
    noisy_count = int(len(sample_indices) * noise_ratio)
    changed = []
    if noisy_count <= 0:
        return noisy_labels, changed

    selected = set(rng.choice(sample_indices, noisy_count, replace=False).tolist())
    for idx in sample_indices:
        if int(idx) not in selected:
            continue

        original = int(noisy_labels[idx])
        if noise_type == "symmetric":
            choices = [label for label in range(num_classes) if label != original]
            new_label = int(rng.choice(choices))
        elif noise_type == "asymmetric":
            new_label = (original + 1) % num_classes
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        noisy_labels[idx] = new_label
        changed.append({"index": int(idx), "original_label": original, "new_label": new_label})

    return noisy_labels, changed


def create_staged_tensor_protocol(
    dataset_name,
    train_data,
    train_labels,
    test_data,
    test_labels,
    num_classes,
    noise_type="symmetric",
    noise_ratio=0.2,
    num_versions=4,
    retention_ratios=None,
    split_ratio=0.5,
    rehearsal_ratio=0.1,
    balanced=False,
    seed=42,
    metadata=None,
):
    if retention_ratios is None:
        retention_ratios = [0.5, 0.3, 0.1, 0.05]
    if len(retention_ratios) < num_versions:
        raise ValueError("retention_ratios must contain at least num_versions values.")

    rng = np.random.default_rng(seed)
    case = settings.get_case(noise_ratio, noise_type, balanced)

    train_data = np.asarray(train_data)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_data = np.asarray(test_data)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    initial_data, initial_labels, inc_data, inc_labels = (
        _balanced_initial_incremental_split(
            train_data, train_labels, num_classes, split_ratio, rng
        )
    )
    aux_data, aux_labels = _sample_rehearsal(
        initial_data, initial_labels, num_classes, rehearsal_ratio, rng
    )

    _save_dataset_file(dataset_name, None, "train_data", train_data)
    _save_dataset_file(dataset_name, None, "train_label", train_labels)
    _save_dataset_file(dataset_name, None, "test_data", test_data)
    _save_dataset_file(dataset_name, None, "test_label", test_labels)
    _save_dataset_file(dataset_name, None, "train_0_data", initial_data)
    _save_dataset_file(dataset_name, None, "train_0_label", initial_labels)
    _save_dataset_file(dataset_name, None, "inc_data", inc_data)
    _save_dataset_file(dataset_name, None, "inc_label", inc_labels)
    _save_dataset_file(dataset_name, None, "aux_data", aux_data)
    _save_dataset_file(dataset_name, None, "aux_label", aux_labels)

    _save_dataset_file(dataset_name, case, "test_data", test_data)
    _save_dataset_file(dataset_name, case, "test_label", test_labels)
    _save_dataset_file(dataset_name, case, "aux_data", aux_data)
    _save_dataset_file(dataset_name, case, "aux_label", aux_labels)
    _save_dataset_file(dataset_name, case, "train_data", initial_data, step=0)
    _save_dataset_file(dataset_name, case, "train_label", initial_labels, step=0)

    forget_classes = set(range(max(1, num_classes // 2)))
    noise_classes = set(range(max(1, num_classes // 2), num_classes))
    if not noise_classes:
        noise_classes = set(range(num_classes))

    forget_indices = [i for i, label in enumerate(inc_labels) if int(label) in forget_classes]
    noise_indices = [i for i, label in enumerate(inc_labels) if int(label) in noise_classes]
    noise_log = []

    for step in range(1, num_versions + 1):
        retention_ratio = retention_ratios[step - 1]
        retain_count = int(len(forget_indices) * retention_ratio)
        retained_forget = (
            rng.choice(forget_indices, retain_count, replace=False)
            if retain_count > 0
            else np.asarray([], dtype=np.int64)
        )
        stage_indices = np.concatenate([retained_forget, np.asarray(noise_indices)])
        stage_labels = inc_labels[stage_indices].copy()
        relative_noise_indices = np.arange(len(retained_forget), len(stage_indices))
        stage_labels, changed = _inject_noise(
            stage_labels,
            relative_noise_indices,
            noise_type,
            noise_ratio,
            num_classes,
            rng,
        )

        permutation = rng.permutation(len(stage_indices))
        stage_data = inc_data[stage_indices][permutation]
        stage_labels = stage_labels[permutation]

        _save_dataset_file(dataset_name, case, "train_data", stage_data, step=step)
        _save_dataset_file(dataset_name, case, "train_label", stage_labels, step=step)
        noise_log.append({"step": step, "changed": changed})

    metadata_path = os.path.join(
        settings.root_dir, "data", dataset_name, "gen", case, "metadata.json"
    )
    _ensure_parent(metadata_path)
    payload = {
        "dataset": dataset_name,
        "case": case,
        "num_classes": int(num_classes),
        "noise_type": noise_type,
        "noise_ratio": float(noise_ratio),
        "num_versions": int(num_versions),
        "retention_ratios": [float(value) for value in retention_ratios],
        "split_ratio": float(split_ratio),
        "rehearsal_ratio": float(rehearsal_ratio),
        "seed": int(seed),
        "metadata": metadata or {},
        "noise_log": noise_log,
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return case
