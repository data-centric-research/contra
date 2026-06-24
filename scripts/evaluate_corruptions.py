import argparse
import copy
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.evaluate_checkpoint import build_parser, evaluate_checkpoint


def metric_value(result):
    return result["mAP"] if "mAP" in result else result["global"]


def main():
    parser = build_parser()
    parser.description = "Evaluate clean and corrupted request streams for one checkpoint."
    parser.add_argument(
        "--corruptions",
        nargs="+",
        default=["gaussian_noise", "gaussian_blur", "jpeg", "contrast"],
    )
    parser.add_argument("--corruption_prefix", default="test")
    parser.add_argument("--summary_json", default=None)
    args = parser.parse_args()

    clean_args = copy.copy(args)
    clean_args.data_name = "test_data"
    clean_args.label_name = "test_label"
    clean_result = evaluate_checkpoint(clean_args)
    clean_metric = metric_value(clean_result)

    summary = {
        "clean": clean_result,
        "corruptions": {},
        "mean_corrupted_to_clean": None,
    }
    ratios = []
    for corruption in args.corruptions:
        corrupt_args = copy.copy(args)
        corrupt_args.data_name = f"{args.corruption_prefix}_{corruption}_data"
        corrupt_args.label_name = f"{args.corruption_prefix}_{corruption}_label"
        result = evaluate_checkpoint(corrupt_args)
        ratio = metric_value(result) / clean_metric if clean_metric > 0 else 0.0
        result["corrupted_to_clean"] = ratio
        summary["corruptions"][corruption] = result
        ratios.append(ratio)

    if ratios:
        summary["mean_corrupted_to_clean"] = sum(ratios) / len(ratios)

    if args.summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
