import os


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))

dataset_paths = {
    "cifar-10": os.path.join(root_dir, "data", "cifar-10"),
    "cifar-100": os.path.join(root_dir, "data", "cifar-100"),
    "food-101": os.path.join(root_dir, "data", "food-101"),
    "pet-37": os.path.join(root_dir, "data", "pet-37"),
    "webvision": os.path.join(root_dir, "data", "webvision"),
}

num_classes_dict = {
    "cifar-10": 10,
    "cifar-100": 100,
    "food-101": 101,
    "pet-37": 37,
    "webvision": 1000,
}

cifar10_config = {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]}

cifar100_config = {"mean": [0.5071, 0.4865, 0.4409], "std": [0.2673, 0.2564, 0.2762]}

food101_config = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

conf_name = "cvpr"


def get_case(noise_ratio, noise_type, suffix=conf_name):
    if isinstance(suffix, bool):
        suffix = "balanced" if suffix else conf_name
    return f"nr_{noise_ratio}_nt_{noise_type}_{suffix}"


def get_num_classes(dataset, override=None):
    if override is not None:
        return int(override)
    try:
        return num_classes_dict[dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset type: {dataset}") from exc


def get_ckpt_path(dataset, case, model, model_suffix, step=None, unique_name=None):
    """Generate and return model paths dynamically."""
    path = os.path.join(root_dir, "ckpt", dataset, case)
    if step is not None and step >= 0:
        path = os.path.join(path, f"step_{step}")
    if unique_name is not None:
        path = os.path.join(path, unique_name)

    return os.path.join(path, f"{model}_{model_suffix}.pth")


# get ckpt files for sensitivity experiment models
def get_pretrain_ckpt_path(
    dataset, case, model, model_suffix, step=None, unique_name=None
):
    """Generate and return model paths dynamically."""
    path = os.path.join(root_dir, "ckpt", dataset, case)
    pretrain_case = "pretrain"
    path = os.path.join(path, pretrain_case)
    if step is not None and step >= 0:
        path = os.path.join(path, f"step_{step}")
    if unique_name is not None:
        path = os.path.join(path, unique_name)

    return os.path.join(path, f"{model}_{model_suffix}.pth")


def get_dataset_path(dataset, case, type, step=None):
    """Generate and return model paths dynamically."""
    dataset_root = dataset_paths.get(dataset, os.path.join(root_dir, "data", dataset))
    path = os.path.join(dataset_root, "gen")
    if case is not None:
        path = os.path.join(path, case)
    if step is not None and step >= 0:
        path = os.path.join(path, f"step_{step}")

    return os.path.join(path, f"{type}.npy")
