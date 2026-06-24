import argparse
import io
import os
import sys

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs import settings


STATS = {
    "none": (None, None),
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "cifar10": ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    "cifar100": ([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]),
}

DATASET_STATS = {
    "cifar-10": "cifar10",
    "cifar-100": "cifar100",
    "pet-37": "imagenet",
    "food-101": "imagenet",
    "webvision": "imagenet",
}


def _stats_for_dataset(dataset, stats_name):
    if stats_name == "auto":
        stats_name = DATASET_STATS.get(dataset, "imagenet")
    mean, std = STATS[stats_name]
    if mean is None:
        return None, None
    return (
        np.asarray(mean, dtype=np.float32).reshape(1, 1, 3),
        np.asarray(std, dtype=np.float32).reshape(1, 1, 3),
    )


def _to_hwc(image):
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image tensor, got shape {image.shape}")
    if image.shape[0] == 3:
        return np.transpose(image, (1, 2, 0)), True
    if image.shape[-1] == 3:
        return image, False
    raise ValueError(f"Cannot locate RGB channel dimension in shape {image.shape}")


def _from_hwc(image, channel_first):
    if channel_first:
        return np.transpose(image, (2, 0, 1))
    return image


def _to_pixel_space(image, mean, std):
    image = image.astype(np.float32, copy=False)
    hwc, channel_first = _to_hwc(image)
    if mean is not None:
        hwc = (hwc * std) + mean
    return np.clip(hwc, 0.0, 1.0), channel_first


def _from_pixel_space(image, channel_first, mean, std, dtype):
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    if mean is not None:
        image = (image - mean) / std
    return _from_hwc(image, channel_first).astype(dtype, copy=False)


def _as_pil(image):
    array = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def _from_pil(image):
    return np.asarray(image).astype(np.float32) / 255.0


def _corrupt_pixel_image(image, corruption, rng, args):
    if corruption == "gaussian_noise":
        return image + rng.normal(0.0, args.noise_std, size=image.shape).astype(np.float32)
    if corruption == "gaussian_blur":
        return _from_pil(_as_pil(image).filter(ImageFilter.GaussianBlur(args.blur_radius)))
    if corruption == "jpeg":
        buffer = io.BytesIO()
        _as_pil(image).save(buffer, format="JPEG", quality=args.jpeg_quality)
        buffer.seek(0)
        return _from_pil(Image.open(buffer).convert("RGB"))
    if corruption == "contrast":
        return _from_pil(ImageEnhance.Contrast(_as_pil(image)).enhance(args.contrast_factor))
    raise ValueError(f"Unsupported corruption: {corruption}")


def corrupt_array(data, corruption, dataset, stats_name, seed, args):
    mean, std = _stats_for_dataset(dataset, stats_name)
    rng = np.random.default_rng(seed)
    output = []
    for image in data:
        pixel_image, channel_first = _to_pixel_space(image, mean, std)
        corrupted = _corrupt_pixel_image(pixel_image, corruption, rng, args)
        output.append(_from_pixel_space(corrupted, channel_first, mean, std, data.dtype))
    return np.stack(output)


def main():
    parser = argparse.ArgumentParser(
        description="Generate corrupted request-stream tensor files from existing staged .npy data."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--noise_ratio", type=float, default=0.2)
    parser.add_argument("--noise_type", choices=["symmetric", "asymmetric"], default="symmetric")
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--input_data_name", default="test_data")
    parser.add_argument("--input_label_name", default="test_label")
    parser.add_argument("--output_prefix", default="test")
    parser.add_argument(
        "--corruptions",
        nargs="+",
        default=["gaussian_noise", "gaussian_blur", "jpeg", "contrast"],
        choices=["gaussian_noise", "gaussian_blur", "jpeg", "contrast"],
    )
    parser.add_argument("--stats", choices=["auto", "none", "imagenet", "cifar10", "cifar100"], default="auto")
    parser.add_argument("--noise_std", type=float, default=0.08)
    parser.add_argument("--blur_radius", type=float, default=1.5)
    parser.add_argument("--jpeg_quality", type=int, default=35)
    parser.add_argument("--contrast_factor", type=float, default=0.55)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    case = settings.get_case(args.noise_ratio, args.noise_type, args.balanced)
    data_path = settings.get_dataset_path(args.dataset, case, args.input_data_name)
    label_path = settings.get_dataset_path(args.dataset, case, args.input_label_name)
    data = np.load(data_path)
    labels = np.load(label_path)
    if args.max_samples is not None:
        data = data[: args.max_samples]
        labels = labels[: args.max_samples]

    for corruption in args.corruptions:
        corrupted = corrupt_array(data, corruption, args.dataset, args.stats, args.seed, args)
        data_name = f"{args.output_prefix}_{corruption}_data"
        label_name = f"{args.output_prefix}_{corruption}_label"
        out_data_path = settings.get_dataset_path(args.dataset, case, data_name)
        out_label_path = settings.get_dataset_path(args.dataset, case, label_name)
        os.makedirs(os.path.dirname(out_data_path), exist_ok=True)
        np.save(out_data_path, corrupted)
        np.save(out_label_path, labels)
        print(f"Saved {corruption}: {out_data_path}")


if __name__ == "__main__":
    main()
