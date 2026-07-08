import os
import numpy as np

from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from core_model.optimizer import create_optimizer_scheduler
import json
from core_model.dataset import BaseTensorDataset

from torch.utils.data import DataLoader

from torchvision.transforms import v2


class TrainTestUtils:
    def __init__(self, model_name, dataset_name):
        self.model_name = model_name
        self.dataset_name = dataset_name

    def create_save_path(self, condition):
        save_dir = os.path.join("models", self.model_name, self.dataset_name, condition)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def l1_regularization(self, model):
        params_vec = []
        for param in model.parameters():
            params_vec.append(param.view(-1))
        return torch.linalg.norm(torch.cat(params_vec), ord=1)

    def train_and_save(
        self,
        model,
        train_loader,
        criterion,
        optimizer,
        save_path,
        epoch,
        num_epochs,
        save_final_model_only=True,
        **kwargs,
    ):
        """
        :param save_final_model_only: If True, only save the model after the final epoch.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        # Extract possible parameters such as alpha or others from kwargs
        alpha = kwargs.get("alpha", 1.0)
        beta = kwargs.get("beta", 0.5)

        with tqdm(
            total=len(train_loader), desc=f"Epoch {epoch + 1} Training", ascii=True
        ) as pbar:
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, labels) * alpha
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)

        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy * 100:.2f}%"
        )

        if not save_final_model_only or epoch == (num_epochs - 1):
            torch.save(
                model.state_dict(),
                os.path.join(
                    save_path, f"{self.model_name}_{self.dataset_name}_final.pth"
                ),
            )
            print(
                f"Final model saved to {os.path.join(save_path, f'{self.model_name}_{self.dataset_name}_final.pth')}"
            )

    def test(self, model, test_loader, condition, progress_bar=None):
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        correct = 0
        total = 0
        running_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if progress_bar:
                    progress_bar.update(1)

        accuracy = correct / total
        avg_loss = running_loss / len(test_loader)
        print(f"Test Accuracy: {100 * accuracy:.2f}%, Loss: {avg_loss:.4f}")

        result = {"accuracy": accuracy, "loss": avg_loss}
        save_dir = os.path.join(
            "results", self.model_name, self.dataset_name, condition
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "performance.json")
        with open(save_path, "w") as f:
            json.dump(result, f)

        print(f"Performance saved to {save_path}")

        return accuracy


def test_model(model, test_loader, criterion, device, epoch):
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        with tqdm(
            total=len(test_loader), desc=f"Epoch {epoch + 1} Testing", ascii=True
        ) as pbar:
            for test_inputs, test_targets in test_loader:
                test_inputs, test_targets = test_inputs.to(device), test_targets.to(
                    device
                )
                test_outputs = model(test_inputs)
                loss = criterion(test_outputs, test_targets)
                test_loss += loss.item()
                _, predicted_test = torch.max(test_outputs, 1)
                total_test += test_targets.size(0)
                correct_test += (predicted_test == test_targets).sum().item()

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct_test / total_test
    print(f"Test Accuracy after Epoch {epoch + 1}: {test_accuracy:.2f}%")
    return test_accuracy, test_loss


def split_train_validation(data, labels, validation_ratio, seed):
    if validation_ratio <= 0:
        return data, labels, None, None

    rng = np.random.default_rng(seed)
    labels_np = np.asarray(labels)
    train_indices, val_indices = [], []
    for label in sorted(set(labels_np.tolist())):
        class_indices = np.flatnonzero(labels_np == label)
        if len(class_indices) <= 1:
            train_indices.extend(class_indices.tolist())
            continue
        shuffled = rng.permutation(class_indices)
        val_count = max(1, int(len(shuffled) * validation_ratio))
        val_count = min(val_count, len(shuffled) - 1)
        val_indices.extend(shuffled[:val_count].tolist())
        train_indices.extend(shuffled[val_count:].tolist())

    if not val_indices:
        return data, labels, None, None

    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)
    return data[train_indices], labels[train_indices], data[val_indices], labels[val_indices]


def train_model(
    model,
    num_classes,
    data,
    labels,
    test_data,
    test_labels,
    epochs=50,
    batch_size=256,
    optimizer_type="adam",
    learning_rate=0.001,
    weight_decay=5e-4,
    data_aug=False,
    dataset_name=None,
    use_early_stopping=False,
    early_stopping_patience=10,
    early_stopping_accuracy_threshold=0.95,
    validation_ratio=0.1,
    seed=42,
    writer=None,
):
    """
    Train model function
    :param model: The ResNet model to be trained
    :param data: The input dataset
    :param labels: The input data labels
    :param test_data: The test set data
    :param test_labels: The test set labels
    :param epochs: The number of training epochs
    :param batch_size: The batch size
    :param optimizer_type: The optimizer
    :param learning_rate: The learning rate
    :return: The trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    criterion = nn.CrossEntropyLoss()

    optimizer, scheduler = create_optimizer_scheduler(
        optimizer_type=optimizer_type,
        parameters=model.parameters(),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        eta_min=0.01 * learning_rate,
    )

    transform_train = None
    transform_test = transforms.Compose(
        [
            # weights.transforms()
        ]
    )
    val_loader = None
    if use_early_stopping:
        data, labels, val_data, val_labels = split_train_validation(
            data,
            labels,
            validation_ratio=validation_ratio,
            seed=seed,
        )
        if val_data is not None:
            val_dataset = BaseTensorDataset(val_data, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataset = BaseTensorDataset(data, labels, transform_train)
    dataloader = DataLoader(
        # dataset, batch_size=batch_size, drop_last=True, shuffle=True
        dataset,
        batch_size=batch_size,
        # drop_last=True,
        drop_last=False,
        shuffle=True,
    )

    test_dataset = BaseTensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    test_accuracies = []

    if data_aug:
        alpha = 0.65
        cutmix_transform = v2.CutMix(alpha=alpha, num_classes=num_classes)
        mixup_transform = v2.MixUp(alpha=alpha, num_classes=num_classes)

    best_state, best_accuracy, stale_epochs = None, -1.0, 0
    eval_loader = val_loader if val_loader is not None else test_loader
    eval_name = "Validation" if val_loader is not None else "Test"

    for epoch in tqdm(range(epochs), desc="Training Progress", ascii=True):
        running_loss = 0.0
        correct = 0
        total = 0

        scheduler.step(epoch)
        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        print("Current LR:", lr)

        with tqdm(
            total=len(dataloader), desc=f"Epoch {epoch + 1} Training", ascii=True
        ) as pbar:
            for inputs, targets in dataloader:
                if len(targets) == 1:
                    inputs = torch.cat([inputs, inputs], dim=0)
                    targets = torch.cat([targets, targets], dim=0)

                targets = targets.to(torch.long)

                if data_aug:
                    transform = mixup_transform  # np.random.choice([mixup_transform, cutmix_transform])
                    inputs, targets = transform(inputs, targets)

                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                mixed_max = torch.argmax(targets.data, 1) if data_aug else targets
                total += targets.size(0)
                correct += (predicted == mixed_max).sum().item()

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)

        avg_loss = running_loss / len(dataloader)
        accuracy = correct / total
        train_losses.append(avg_loss)
        print(
            f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy * 100:.2f}%"
        )

        if writer:
            writer.add_scalar("Train/Loss", avg_loss, epoch)
            writer.add_scalar("Train/Accuracy", accuracy * 100, epoch)

        test_accuracy, test_loss = test_model(model, eval_loader, criterion, device, epoch)
        test_accuracies.append(test_accuracy)

        if writer:
            writer.add_scalar(f"{eval_name}/Loss", test_loss, epoch)
            writer.add_scalar(f"{eval_name}/Accuracy", test_accuracy, epoch)

        if use_early_stopping:
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                stale_epochs = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                stale_epochs += 1

            if test_accuracy >= early_stopping_accuracy_threshold * 100:
                print(f"Early stopping: reached {eval_name.lower()} threshold.")
                break
            if stale_epochs >= early_stopping_patience:
                print(f"Early stopping: no {eval_name.lower()} improvement.")
                break

        model.train()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
