import os
import argparse

from core_model.custom_model import SUPPORTED_MODELS

# from run_experiment import run_experiment


# Custom validation helpers.
def check_positive(value):
    ivalue = float(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive float value")
    return ivalue


def parse_kwargs(kwargs):
    """
    Parse --kwargs entries in key=value form into a dictionary.
    """
    parsed_kwargs = {}
    if kwargs:
        for kwarg in kwargs:
            key, value = kwarg.split("=")
            parsed_kwargs[key] = (
                float(value) if "." in value else int(value)
            )  # Convert values based on their literal form.
    return parsed_kwargs


def make_arg_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
        description="Run CONTRA experiments with different datasets, models, and noise settings."
        )

    # Dataset and model options.
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="cifar-10",
        choices=[
            "cifar-10",
            "cifar-100",
            "pet-37",
            "food-101",
            "webvision",
        ],
        help="Dataset name. Choose from: cifar-10, cifar-100, pet-37, food-101, webvision.",
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Override the dataset class count, useful for compact WebVision subsets.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=SUPPORTED_MODELS,
        help="Model backbone.",
    )

    # Model initialization and training options.
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet-pretrained weights for torchvision backbones; disable with --no-pretrained.",
    )

    parser.add_argument(
        "--no_spnorm",
        action="store_true",
        default=False,
        help="If specified, no spectral norm",
    )

    parser.add_argument(
        "--data_aug",
        action="store_true",
        default=False,
        help="If specified, do data augmentation",
    )

    # Label-noise configuration.
    parser.add_argument(
        "--noise_type",
        type=str,
        choices=["symmetric", "asymmetric"],
        default="symmetric",
        help="Noise type to use, e.g., symmetric or asymmetric",
    )

    parser.add_argument(
        "--balanced",
        default=False,
        action="store_true",
        help="Use the balanced case name used in the experiments.",
    )

    parser.add_argument(
        "--train_aux",
        default=False,
        action="store_true",
        help="Training with auxiliary dataset",
    )

    parser.add_argument(
        "--tta_only",
        default=None,
        type=int,
        choices=[0, 1],
        help="Ablation mode: 0 runs adaptation without prior repair; 1 runs adaptation from saved repair checkpoints.",
    )

    parser.add_argument(
        "--eval_map",
        action="store_true",
        default=False,
        help="Also report retrieval mAP from penultimate-layer embeddings.",
    )

    parser.add_argument(
        "--map_batch_size",
        type=int,
        default=512,
        help="Batch size used for cosine-similarity retrieval mAP evaluation.",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Continual learning step",
    )

    parser.add_argument(
        "--train_mode",
        type=str,
        choices=["pretrain", "inc_train", "finetune", "retrain", "train"],
        help="Train mode",
    )

    parser.add_argument(
        "--noise_ratio",
        type=float,
        default=0.2,
        help="Noise ratio",
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="Specify the GPU(s) to use, e.g., --gpu 0,1 for multi-GPU or --gpu 0 for single GPU",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits, training, and batch scripts.",
    )

    # Optimization options.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training (default: 128; matches ICANN manuscript)",
    )

    parser.add_argument(
        "--learning_rate",
        type=check_positive,  # Use the custom validation helper.
        default=0.001,
        help="Learning rate for the optimizer (default: 0.001)",
    )

    parser.add_argument(
        "--teacher_lr_scale",
        type=check_positive,  # Use the custom validation helper.
        default=0.2,
        help="Teacher learning rate scale",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adam"],
        default="adam",
        help="Optimizer for training weights",
    )

    # Momentum for SGD.
    parser.add_argument(
        "--momentum",
        type=check_positive,
        default=0.9,
        help="Momentum for SGD optimizer (default: 0.9). Only used if optimizer is 'sgd'.",
    )

    # Weight decay.
    parser.add_argument(
        "--weight_decay",
        type=check_positive,
        default=5e-4,
        help="Weight decay for the optimizer (default: 0.0005).",
    )

    # Epoch and stopping options.
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Number of epochs to train the model (default: 200)",
    )

    parser.add_argument(
        "--ul_epochs",
        type=int,
        default=3,
        help="Number of label-refinement epochs",
    )

    parser.add_argument(
        "--agree_epochs",
        type=int,
        default=2,
        help="Number of agreement-filtering epochs",
    )

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Patience for early stopping (default: 10)",
    )

    parser.add_argument(
        "--early_stopping_accuracy_threshold",
        type=float,
        default=0.95,
        help="Accuracy threshold for early stopping (default: 0.95)",
    )

    # Early-stopping switch.
    parser.add_argument(
        "--use_early_stopping",
        action="store_true",
        help="Enable early stopping if specified, otherwise train for the full number of epochs",
    )

    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.1,
        help="Class-balanced training split ratio used as validation when early stopping is enabled.",
    )

    # Number of repair-training iterations.
    parser.add_argument(
        "--repair_iter_num",
        type=int,
        default=3,
        help="The number of iterations to train the model",
    )

    # Number of adaptation iterations.
    parser.add_argument(
        "--adapt_iter_num",
        type=int,
        default=3,
        help="The number of iterations to adapt the model",
    )

    parser.add_argument(
        "--adapt_epochs",
        type=int,
        default=1,
        help="The number of epochs to adapt the model",
    )

    parser.add_argument(
        "--lr_scale",
        type=float,
        default=0.5,
        help="Scale the working model lr",
    )

    parser.add_argument(
        "--ls_gamma",
        type=float,
        default=0.25,
        help="Label smoothing factor",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.75,
        help="Sharpen factor",
    )

    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.4,
        help="Mixup alpha (default: 0.4; matches ICANN manuscript)",
    )

    parser.add_argument(
        "--centroid_ratio",
        type=float,
        default=0.1,
        help="Per-class nearest-centroid ratio for confident teacher guidance.",
    )

    parser.add_argument(
        "--conf_ratio",
        type=float,
        default=0.1,
        help="Top-confidence agreement ratio retained as TTA references.",
    )

    parser.add_argument(
        "--student_spnorm",
        action="store_true",
        help="Ablation: also apply spectral normalization to the student branch.",
    )

    parser.add_argument(
        "--disable_agreement",
        action="store_true",
        help="Ablation: remove agreement samples D_a from repair and TTA references.",
    )

    parser.add_argument(
        "--disable_centroid",
        action="store_true",
        help="Ablation: remove nearest-centroid disagreement samples from repair.",
    )

    parser.add_argument(
        "--disable_mixup",
        action="store_true",
        help="Ablation: train on unmixed samples instead of Mixup pairs.",
    )

    # Additional keyword arguments.
    parser.add_argument(
        "--kwargs", nargs="*", help="Additional key=value arguments for hyperparameters"
    )

    # Optional model name suffix.
    parser.add_argument(
        "--model_suffix",
        type=str,
        default=None,
        help="Suffix to save model name",
    )

    parser.add_argument("--uni_name", type=str, default=None, help="Model unique name")

    parser.add_argument(
        "--use_tensorboard", action="store_true", help="Use TensorBoard for logging."
    )

    return parser


# Command-line parsing.
def parse_args():
    parser = make_arg_parser()
    # Return parsed arguments.
    return parser.parse_args()


def main():
    # Parse command-line arguments.
    args = parse_args()

    # Set the GPU environment variable.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"Using GPU(s): {args.gpu}")

    # Parse extra keyword arguments.
    kwargs = parse_kwargs(args.kwargs)

    # Print the selected configuration.
    print(f"Running experiment with the following configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Num Classes Override: {args.num_classes}")
    print(f"  Model: {args.model}")
    print(f"  Seed: {args.seed}")
    print(f"  Pretrained: {args.pretrained}")
    print(f"  Noise Type: {args.noise_type}")
    print(f"  Noise ratio: {args.noise_ratio}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Optimizer: {args.optimizer}")
    if args.optimizer == "sgd":
        print(f"  Momentum: {args.momentum}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Number of Epochs: {args.num_epochs}")
    print(f"  Eval mAP: {args.eval_map}")
    print(f"  Use Early Stopping: {args.use_early_stopping}")
    if args.use_early_stopping:
        print(f"  Early Stopping Patience: {args.early_stopping_patience}")
        print(
            f"  Early Stopping Accuracy Threshold: {args.early_stopping_accuracy_threshold}"
        )
    print(f"  Repair Iterations: {args.repair_iter_num}")
    print(f"  Adaptation Iterations: {args.adapt_iter_num}")
    print(f"  Centroid Ratio: {args.centroid_ratio}")
    print(f"  Confidence Ratio: {args.conf_ratio}")

    print(f"Additional kwargs: {kwargs}")


if __name__ == "__main__":
    main()
