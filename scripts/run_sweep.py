import argparse
import shlex
import subprocess
import sys


GENERATOR_BY_DATASET = {
    "cifar-10": "gen_dataset/gen_cifar10_exp_data.py",
    "cifar-100": "gen_dataset/gen_cifar100_exp_data.py",
    "pet-37": "gen_dataset/gen_pet37_exp_data.py",
    "food-101": "gen_dataset/gen_imagefolder_exp_data.py",
    "webvision": "gen_dataset/gen_webvision_exp_data.py",
}


def command_to_text(command):
    return " ".join(shlex.quote(part) for part in command)


def common_args(args, step, value):
    values = [
        "--step", str(step),
        "--model", args.model,
        "--dataset", args.dataset,
        "--noise_ratio", str(args.noise_ratio if args.preset != "noise_ratio" else value),
        "--noise_type", args.noise_type,
        "--num_epochs", str(args.num_epochs),
        "--learning_rate", str(args.learning_rate),
        "--optimizer", args.optimizer,
        "--batch_size", str(args.batch_size),
        "--seed", str(args.seed),
        "--uni_name", f"sweep_{args.preset}_{value}",
    ]
    if args.balanced:
        values.append("--balanced")
    if args.num_classes is not None:
        values.extend(["--num_classes", str(args.num_classes)])
    if args.eval_map:
        values.append("--eval_map")
    return values


def without_uni_name(values):
    output = []
    skip_next = False
    for value in values:
        if skip_next:
            skip_next = False
            continue
        if value == "--uni_name":
            skip_next = True
            continue
        output.append(value)
    return output


def generator_command(args, value):
    generator = GENERATOR_BY_DATASET[args.dataset]
    command = [
        args.python,
        generator,
        "--noise_type",
        args.noise_type,
        "--noise_ratio",
        str(args.noise_ratio if args.preset != "noise_ratio" else value),
        "--seed",
        str(args.seed),
    ]
    if args.balanced:
        command.append("--balanced")
    if args.preset == "rehearsal_ratio":
        command.extend(["--rehearsal_ratio", str(value)])
    if args.dataset in {"food-101", "webvision"}:
        if args.train_root:
            command.extend(["--train_root", args.train_root])
        if args.test_root:
            command.extend(["--test_root", args.test_root])
        if args.train_list:
            command.extend(["--train_list", args.train_list])
        if args.test_list:
            command.extend(["--test_list", args.test_list])
        if args.image_root:
            command.extend(["--image_root", args.image_root])
        if args.num_classes is not None:
            command.extend(["--num_classes", str(args.num_classes)])
    else:
        if args.data_dir:
            command.extend(["--data_dir", args.data_dir])
    return command


def contra_command(args, value, step):
    command = [args.python, "run_contra.py", *common_args(args, step, value)]
    if args.preset == "mixup_alpha":
        command.extend(["--mixup_alpha", str(value)])
    elif args.preset == "centroid_ratio":
        command.extend(["--centroid_ratio", str(value)])
    elif args.preset == "conf_ratio":
        command.extend(["--conf_ratio", str(value)])
    elif args.preset == "adapt_iter_num":
        command.extend(["--adapt_iter_num", str(int(value))])
    return command


def eval_command(args, value):
    return [
        args.python,
        "scripts/evaluate_checkpoint.py",
        *common_args(args, args.step, value),
        "--model_suffix",
        "worker_tta",
    ]


def main():
    parser = argparse.ArgumentParser(description="Run or print hyper-parameter sweep commands.")
    parser.add_argument(
        "--preset",
        choices=[
            "mixup_alpha",
            "centroid_ratio",
            "conf_ratio",
            "adapt_iter_num",
            "rehearsal_ratio",
            "noise_ratio",
        ],
        required=True,
    )
    parser.add_argument("--values", type=float, nargs="+", required=True)
    parser.add_argument("--dataset", default="pet-37")
    parser.add_argument("--model", default="wideresnet50")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--noise_ratio", type=float, default=0.2)
    parser.add_argument("--noise_type", choices=["symmetric", "asymmetric"], default="symmetric")
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_map", action="store_true")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--train_root", default=None)
    parser.add_argument("--test_root", default=None)
    parser.add_argument("--train_list", default=None)
    parser.add_argument("--test_list", default=None)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    commands = []
    for value in args.values:
        if args.preset in {"rehearsal_ratio", "noise_ratio"}:
            commands.append(generator_command(args, value))
        commands.append(
            [
                args.python,
                "run_experiment.py",
                *without_uni_name(common_args(args, 0, value)),
            ]
        )
        commands.append([args.python, "run_experiment.py", *common_args(args, 0, value)])
        for stage in range(1, args.step + 1):
            commands.append(contra_command(args, value, stage))
        commands.append(eval_command(args, value))

    for command in commands:
        print(command_to_text(command))
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
