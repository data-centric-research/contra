import argparse
import shlex
import subprocess
import sys


ABLATIONS = {
    "full": [],
    "repair_only": ["--adapt_iter_num", "0"],
    "adapt_only": ["--tta_only", "0"],
    "no_spnorm": ["--no_spnorm"],
    "student_spnorm": ["--student_spnorm"],
    "disable_agreement": ["--disable_agreement"],
    "disable_centroid": ["--disable_centroid"],
    "disable_mixup": ["--disable_mixup"],
}


def command_to_text(command):
    return " ".join(shlex.quote(part) for part in command)


def common_args(args, uni_name, step=None):
    if step is None:
        step = args.step
    values = [
        "--step",
        str(step),
        "--model",
        args.model,
        "--dataset",
        args.dataset,
        "--noise_ratio",
        str(args.noise_ratio),
        "--noise_type",
        args.noise_type,
        "--num_epochs",
        str(args.num_epochs),
        "--adapt_epochs",
        str(args.adapt_epochs),
        "--adapt_iter_num",
        str(args.adapt_iter_num),
        "--learning_rate",
        str(args.learning_rate),
        "--optimizer",
        args.optimizer,
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
    ]
    if uni_name is not None:
        values.extend(["--uni_name", uni_name])
    if args.balanced:
        values.append("--balanced")
    if args.num_classes is not None:
        values.extend(["--num_classes", str(args.num_classes)])
    if args.eval_map:
        values.append("--eval_map")
    return values


def eval_command(args, uni_name, suffix):
    command = [
        args.python,
        "scripts/evaluate_checkpoint.py",
        *common_args(args, uni_name),
        "--model_suffix",
        suffix,
    ]
    if uni_name is not None and "student_spnorm" in uni_name:
        command.append("--student_spnorm")
    return command


def commands_for_ablation(args, name):
    if name not in ABLATIONS:
        raise ValueError(f"Unsupported ablation: {name}")

    uni_name = None if name == "adapt_only" else f"ablation_{name}"
    commands = []
    if uni_name is not None:
        commands.append(
            [
                args.python,
                "run_experiment.py",
                *common_args(args, uni_name, step=0),
            ]
        )

    stage_range = [args.step] if name == "adapt_only" else range(1, args.step + 1)
    if name == "adapt_only":
        commands.append(
            [
                args.python,
                "run_experiment.py",
                *common_args(args, uni_name),
                "--model_suffix",
                "worker_raw",
            ]
        )

    for stage in stage_range:
        run_command = [
            args.python,
            "run_contra.py",
            *common_args(args, uni_name, step=stage),
            *ABLATIONS[name],
        ]
        commands.append(run_command)

    suffix = "worker_restore" if name == "repair_only" else "worker_tta"
    commands.append(eval_command(args, uni_name, suffix))
    return commands


def main():
    parser = argparse.ArgumentParser(
        description="Print or run paper ablation commands for CONTRA."
    )
    parser.add_argument(
        "--ablation",
        choices=list(ABLATIONS),
        nargs="+",
        default=list(ABLATIONS),
    )
    parser.add_argument("--dataset", default="pet-37")
    parser.add_argument("--model", default="wideresnet50")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--noise_ratio", type=float, default=0.2)
    parser.add_argument("--noise_type", choices=["symmetric", "asymmetric"], default="symmetric")
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--adapt_epochs", type=int, default=5)
    parser.add_argument("--adapt_iter_num", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_map", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    commands = []
    for ablation in args.ablation:
        commands.extend(commands_for_ablation(args, ablation))

    for command in commands:
        print(command_to_text(command))
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
