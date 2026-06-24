import argparse
import shlex
import subprocess
import sys


def common_args(args, seed, step):
    values = [
        "--step", str(step),
        "--model", args.model,
        "--dataset", args.dataset,
        "--noise_ratio", str(args.noise_ratio),
        "--noise_type", args.noise_type,
        "--num_epochs", str(args.num_epochs),
        "--learning_rate", str(args.learning_rate),
        "--optimizer", args.optimizer,
        "--batch_size", str(args.batch_size),
        "--seed", str(seed),
    ]
    if args.balanced:
        values.append("--balanced")
    if args.num_classes is not None:
        values.extend(["--num_classes", str(args.num_classes)])
    if args.eval_map:
        values.append("--eval_map")
    return values


def command_to_text(command):
    return " ".join(shlex.quote(part) for part in command)


def add_eval(commands, args, seed, step, suffix, uni_name=None):
    command = [
        args.python,
        "scripts/evaluate_checkpoint.py",
        *common_args(args, seed, step),
        "--model_suffix",
        suffix,
    ]
    if uni_name:
        command.extend(["--uni_name", uni_name])
    commands.append(command)


def commands_for_seed(args, seed):
    commands = []
    for step in args.steps:
        if step == 0:
            commands.append([args.python, "run_experiment.py", *common_args(args, seed, step)])
            add_eval(commands, args, seed, step, "worker_restore")
            continue

        method_name = args.method
        if method_name in {"raw", "contra", "rehearsal"}:
            raw_command = [
                args.python,
                "run_experiment.py",
                *common_args(args, seed, step),
                "--model_suffix",
                "worker_raw",
            ]
            if method_name == "rehearsal":
                raw_command.extend(["--uni_name", "Rehearsal"])
            commands.append(raw_command)

        if method_name == "raw":
            add_eval(commands, args, seed, step, "worker_raw")
        elif method_name == "contra":
            commands.append([args.python, "run_contra.py", *common_args(args, seed, step)])
            add_eval(commands, args, seed, step, "worker_tta")
        elif method_name == "rehearsal":
            commands.append(
                [
                    args.python,
                    "run_experiment.py",
                    *common_args(args, seed, step),
                    "--train_aux",
                    "--uni_name",
                    "Rehearsal",
                ]
            )
            add_eval(commands, args, seed, step, "worker_restore", "Rehearsal")
        elif method_name in {"Coteaching", "Coteachingplus", "JoCoR", "DivideMix"}:
            commands.append(
                [
                    args.python,
                    "baseline_code/colearn/main.py",
                    *common_args(args, seed, step),
                    "--uni_name",
                    method_name,
                ]
            )
            add_eval(commands, args, seed, step, "worker_restore", method_name)
        elif method_name == "SoTTA":
            commands.append(
                [
                    args.python,
                    "baseline_code/sotta/run_sotta.py",
                    *common_args(args, seed, step),
                    "--uni_name",
                    "SoTTA",
                ]
            )
            add_eval(commands, args, seed, step, "worker_tta", "SoTTA")
        elif method_name == "CoTTA":
            commands.append(
                [
                    args.python,
                    "baseline_code/cotta-main/cifar/tta.py",
                    *common_args(args, seed, step),
                    "--uni_name",
                    "CoTTA",
                ]
            )
            add_eval(commands, args, seed, step, "worker_tta", "CoTTA")
        elif method_name == "PLF":
            commands.append(
                [
                    args.python,
                    "baseline_code/PLF-main/cifar/run_plf.py",
                    *common_args(args, seed, step),
                    "--uni_name",
                    "PLF",
                ]
            )
            add_eval(commands, args, seed, step, "worker_tta", "PLF")
        else:
            raise ValueError(f"Unsupported method: {method_name}")
    return commands


def main():
    parser = argparse.ArgumentParser(description="Run or print multi-seed experiment commands.")
    parser.add_argument("--method", default="contra")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--dataset", default="pet-37")
    parser.add_argument("--model", default="wideresnet50")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--noise_ratio", type=float, default=0.2)
    parser.add_argument("--noise_type", choices=["symmetric", "asymmetric"], default="symmetric")
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_map", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    commands = []
    for seed in args.seeds:
        commands.extend(commands_for_seed(args, seed))

    for command in commands:
        print(command_to_text(command))
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
