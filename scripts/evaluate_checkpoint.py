import argparse
import json
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from args_paser import make_arg_parser
from configs import settings
from core_model.custom_model import ClassifierWrapper, load_custom_model
from core_model.dataset import get_dataset_loader
from core_model.reproducibility import set_global_seed
from core_model.train_test import model_test


def build_parser():
    parser = make_arg_parser(
        argparse.ArgumentParser(description="Evaluate a saved checkpoint with accuracy and optional retrieval mAP.")
    )
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--data_name", default="test_data")
    parser.add_argument("--label_name", default="test_label")
    parser.add_argument("--json_out", default=None)
    parser.add_argument("--device", default=None)
    return parser


def evaluate_checkpoint(args):
    set_global_seed(args.seed)
    case = settings.get_case(args.noise_ratio, args.noise_type, args.balanced)
    num_classes = settings.get_num_classes(args.dataset, args.num_classes)
    checkpoint_path = args.checkpoint_path or settings.get_ckpt_path(
        args.dataset,
        case,
        args.model,
        args.model_suffix or "worker_restore",
        step=args.step,
        unique_name=args.uni_name,
    )

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    backbone = load_custom_model(args.model, num_classes, load_pretrained=False)
    model = ClassifierWrapper(
        backbone,
        num_classes,
        spectral_norm=getattr(args, "student_spnorm", False),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)

    _, _, data_loader = get_dataset_loader(
        args.dataset,
        "test",
        case,
        None,
        None,
        None,
        batch_size=args.batch_size,
        num_classes=num_classes,
        shuffle=False,
        data_name=args.data_name,
        label_name=args.label_name,
    )
    result = model_test(
        data_loader,
        model,
        device=device,
        compute_map=args.eval_map,
        map_batch_size=args.map_batch_size,
    )
    result["checkpoint_path"] = checkpoint_path
    result["data_name"] = args.data_name
    result["label_name"] = args.label_name

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    return result


def main():
    args = build_parser().parse_args()
    result = evaluate_checkpoint(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
