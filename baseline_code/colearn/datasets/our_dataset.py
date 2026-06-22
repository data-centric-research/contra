from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class CustomDataset(Dataset):
    def __init__(self, data, label, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
        data = data.astype(np.float32)
        shape = data.shape
        channel_idx = np.where(np.array(shape) == 3)[0]
        if channel_idx == 2:
            data = np.transpose(data, [0, 2, 1, 3])
        if channel_idx == 3:
            data = np.transpose(data, [0, 3, 1, 2])

        if (data > 1).any():
            self.data = data / 255.
        else:
            self.data = data

        # normalize mean std
        mean = np.array(mean, dtype=np.float32)[np.newaxis, :, np.newaxis, np.newaxis]
        std = np.array(std, dtype=np.float32)[np.newaxis, :, np.newaxis, np.newaxis]
        self.data = (self.data - mean) / std

        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


def get_dataset_loader(
    dataset_name, loader_name, data_dir, batch_size, num_classes=0, drop_last=False, shuffle=False
):
    """
    Load the dataset selected by loader_name: incremental training (inc),
    auxiliary data (aux), test data (test), or train data (train).
    """
    if loader_name in ['inc', 'aux', 'test', 'train']:
        data_name = "%s_%s_data.npy" % (dataset_name, loader_name)
        label_name = "%s_%s_labels.npy" % (dataset_name, loader_name)
    else:
        raise ValueError(
            f"Invalid loader_name {loader_name}. Choose from 'inc', 'aux', or 'test'."
        )

    data_path = os.path.join(data_dir, data_name)
    label_path = os.path.join(data_dir, label_name)

    # Check whether the required files exist.
    if not os.path.exists(data_path) or not os.path.exists(label_path):
        raise FileNotFoundError(f"{data_name} or {label_name} not found in {data_dir}")

    data = np.load(data_path)
    label = np.load(label_path)
    # if loader_name == 'train':  # train label change to onehot for teacher model
    #     label = np.eye(num_classes)[label]

    # Build the custom dataset.
    dataset = CustomDataset(data, label)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle
    )

    return dataset, data_loader


if __name__ == "__main__":
    # Example CIFAR-10 data directory.
    data_dir = "./data/cifar-10/noise/"
    # data_dir = "../data/cifar-100/noise/"
    batch_size = 32

    # Test loading the incremental dataset.
    print("Loading Incremental Training Dataset (inc)")
    inc_dataset, inc_loader = get_dataset_loader("inc", data_dir, batch_size)
    print(f"Incremental Dataset Size: {len(inc_dataset)}")

    # Inspect one incremental-data batch.
    for images, labels in inc_loader:
        print(f"Batch Image Shape: {images.shape}, Batch Label Shape: {labels.shape}")
        break  # Print only the first batch.

    # Test loading the auxiliary dataset.
    print("\nLoading Auxiliary Dataset (aux)")
    aux_dataset, aux_loader = get_dataset_loader("aux", data_dir, batch_size)
    print(f"Auxiliary Dataset Size: {len(aux_dataset)}")

    # Inspect one auxiliary-data batch.
    for images, labels in aux_loader:
        print(f"Batch Image Shape: {images.shape}, Batch Label Shape: {labels.shape}")
        break

    # Test loading the test dataset.
    print("\nLoading Test Dataset (test)")
    test_dataset, test_loader = get_dataset_loader("test", data_dir, batch_size)
    print(f"Test Dataset Size: {len(test_dataset)}")

    # Inspect one test-data batch.
    for images, labels in test_loader:
        print(f"Batch Image Shape: {images.shape}, Batch Label Shape: {labels.shape}")
        break
