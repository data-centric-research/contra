import importlib
from utils import set_seed, get_test_acc
import algorithms
import numpy as np
import torch
import os

from core_model.dataset import get_dataset_loader
from args_paser import parse_args
from configs import settings
from core_model.custom_model import load_custom_model, ClassifierWrapper
from core_model.reproducibility import set_global_seed


def resolve_lnl_checkpoint(dataset, case, model_name, step, uni_name):
    candidates = [
        settings.get_ckpt_path(
            dataset, case, model_name, "worker_raw", step=step, unique_name=uni_name
        ),
        settings.get_ckpt_path(
            dataset, case, model_name, "worker_raw", step=step, unique_name=None
        ),
    ]
    if step > 0:
        candidates.extend(
            [
                settings.get_ckpt_path(
                    dataset,
                    case,
                    model_name,
                    "worker_restore",
                    step=step - 1,
                    unique_name=uni_name,
                ),
                settings.get_ckpt_path(
                    dataset,
                    case,
                    model_name,
                    "worker_restore",
                    step=step - 1,
                    unique_name="contra",
                ),
                settings.get_ckpt_path(
                    dataset,
                    case,
                    model_name,
                    "worker_restore",
                    step=step - 1,
                    unique_name=None,
                ),
            ]
        )

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "No compatible LNL starting checkpoint found. Tried:\n"
        + "\n".join(candidates)
    )


def main():
    custom_args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = custom_args.gpu
    case = settings.get_case(
        custom_args.noise_ratio, custom_args.noise_type, custom_args.balanced
    )
    step = custom_args.step
    uni_name = custom_args.uni_name
    num_classes = settings.get_num_classes(custom_args.dataset, custom_args.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dynamically load the paper-aligned LNL baseline configuration.
    config_modules = {
        "Coteachingplus": "co_configs.coteachingplus",
        "Coteaching": "co_configs.coteaching",
        "JoCoR": "co_configs.jocor",
        "DivideMix": "co_configs.dividemix",
    }

    if uni_name not in config_modules:
        raise ValueError(f"Unknown uni_name: {uni_name}")

    config_module = importlib.import_module(config_modules[uni_name])
    config = config_module.config
    config["epochs"] = custom_args.num_epochs
    config["batch_size"] = custom_args.batch_size
    config["seed"] = custom_args.seed

    set_seed(config["seed"])
    set_global_seed(config["seed"])

    if config["algorithm"] == "Coteachingplus":
        model = algorithms.Coteachingplus()
        train_mode = "train"
    elif config["algorithm"] == "Coteaching":
        model = algorithms.Coteaching()
        train_mode = "train_single"
    elif config["algorithm"] == "JoCoR":
        model = algorithms.JoCoR()
        train_mode = "train_single"
    elif config["algorithm"] == "DivideMix":
        model = algorithms.DivideMix(config, num_classes=num_classes)
        train_mode = "train_single"
    else:
        raise ValueError(f"Unsupported LNL baseline: {config['algorithm']}")

    _, _, trainloader = get_dataset_loader(
        custom_args.dataset,
        "train",
        case,
        step,
        None,
        None,
        custom_args.batch_size,
        shuffle=True,
    )

    _, _, testloader = get_dataset_loader(
        custom_args.dataset,
        "test",
        case,
        None,
        None,
        None,
        custom_args.batch_size,
        shuffle=False,
    )

    num_test_images = len(testloader.dataset)

    load_model_path = resolve_lnl_checkpoint(
        custom_args.dataset,
        case,
        custom_args.model,
        step,
        uni_name,
    )
    save_model_path = settings.get_ckpt_path(
        custom_args.dataset,
        case,
        custom_args.model,
        model_suffix="worker_restore",
        step=step,
        unique_name=uni_name,
    )
    model.epochs = custom_args.num_epochs

    loaded_model1 = load_custom_model(
        custom_args.model, num_classes, load_pretrained=False
    )
    model.model1 = ClassifierWrapper(loaded_model1, num_classes)

    loaded_model2 = load_custom_model(
        custom_args.model, num_classes, load_pretrained=False
    )
    model.model2 = ClassifierWrapper(loaded_model2, num_classes)

    print("Loading starting checkpoint:", load_model_path)
    checkpoint = torch.load(load_model_path, map_location=device)
    model.model1.load_state_dict(checkpoint, strict=False)
    model.model2.load_state_dict(checkpoint, strict=False)

    model.model1.to(device)
    model.model2.to(device)
    if config["algorithm"] != "DivideMix":
        model.set_optimizer(custom_args.dataset, num_classes, config)

    epoch = 0
    # evaluate models with random weights
    test_acc = get_test_acc(model.evaluate(testloader))
    print(
        "Epoch [%d/%d] Test Accuracy on the %s test images: %.4f"
        % (epoch + 1, custom_args.num_epochs, num_test_images, test_acc)
    )

    acc_list, acc_all_list = [], []
    best_acc, best_epoch = 0.0, 0

    for epoch in range(1, custom_args.num_epochs):
        # train
        model.train(trainloader, epoch)
        # evaluate
        test_acc, test_acc2 = model.evaluate(testloader)
        if best_acc < test_acc:
            best_acc, best_epoch = test_acc, epoch

        print(
            "Epoch [%d/%d] Test Accuracy on the %s test images: %.4f %%"
            % (epoch + 1, custom_args.num_epochs, num_test_images, test_acc)
        )

        if epoch >= custom_args.num_epochs - 10:
            acc_list.extend([test_acc])
        acc_all_list.extend([test_acc])

        # save model1
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        torch.save(model.model1.state_dict(), save_model_path)
        print("model saved to:", save_model_path)

    if config["save_result"]:
        acc_np = np.array(acc_list)
        print(f"Mean accuracy over the final epochs: {acc_np.mean():.4f}")


if __name__ == "__main__":
    main()
