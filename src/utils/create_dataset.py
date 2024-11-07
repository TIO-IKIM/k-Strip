# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
from logging import Logger
import torch
import time
from glob import glob
from torch.utils.data import DataLoader
from pathlib import Path
import utils.transforms as transform


class CreateDataset:
    """
    Custom dataset class for loading data.

    Parameters:
        data_path (str): Path to the data.
        res (bool): Whether to use residual learning. Defaults to False.
        transform (str): Type of data transformation. Defaults to None.
        augmentation (callable): Data augmentation function. Defaults to None.
        config (dict): Configuration dictionary. Defaults to None.
        **kwargs: Additional keyword arguments.

    Attributes:
        data_path (str): Path to the data.
        config (dict): Configuration dictionary.
        res (bool): Whether to use residual learning.
        transform (str): Type of data transformation.
        augmentation (callable): Data augmentation function.
        file_names (list): List of file names in the dataset.

    Methods:
        __len__: Get the number of examples in the dataset.
        __load_tensor: Load a tensor from a given path.
        __getitem__: Get a specific item from the dataset.
        standardization: Perform standardization on the data.
        scaling: Perform scaling on the data.
    """

    def __init__(
        self,
        data_path: str,
        transform=None,
        augmentation=None,
        config=None,
        **kwargs,
    ):
        """
        Initialize the CreateDataset instance.
        """
        self.data_path = data_path
        self.config = config
        self.transform = transform

        self.file_names = []

        for file in glob(f"{self.data_path}/orig/*"):
            self.file_names.append(Path(file).name)

        self.augmentation = augmentation

    def __len__(self):
        """
        Get the number of examples in the dataset.
        """
        return len(self.file_names)

    def __load_tensor(self, path):
        """
        Load a tensor from a given path.
        """
        return torch.load(path, map_location="cpu", weights_only=True)

    def __getitem__(self, idx):
        """
        Get a specific item from the dataset.

        Parameters:
            idx (int): Index of the item.

        Returns:
            tuple: A tuple containing the input tensor (self.x) and the target tensor (self.y).
        """
        self.x = self.__load_tensor(
            Path(self.data_path) / "orig" / self.file_names[idx]
        )
        self.y = self.__load_tensor(
            Path(self.data_path) / "label" / self.file_names[idx]
        )

        self.x = (self.x).cfloat()
        self.y = (self.y).cfloat()
        self.x = self.x[None, ...]
        self.y = self.y[None, ...]
        if torch.isnan(self.x).any() or torch.isnan(self.y).any():
            self.x[torch.isnan(self.x)] = 0
            self.y[torch.isnan(self.y)] = 0

        if self.augmentation:
            self.x, self.y = self.augmentation(self.x, self.y)

        if self.transform == "standardization":
            CreateDataset.standardization(self)
        if self.transform == "scaling":
            CreateDataset.scaling(self)
        if self.transform == "positive":
            self.x = transform.positive(self.x)

        if self.x.ndim == self.y.ndim and self.x.ndim not in [3, 4]:
            raise ValueError(
                f"Data needs to be 2D or 3D, but has {self.x.ndim - 1} dimensions!"
            )

        return self.x, self.y

    def standardization(self) -> None:
        """
        Perform standardization on the data.
        """
        y_std = self.y.std()
        x_std = self.x.std()
        
        if x_std != 0:
            self.x = ((self.x.abs() - self.x.mean()) / x_std) * torch.exp(1j * self.x.angle())
        if y_std != 0:
            self.y = ((self.y.abs() - self.y.mean()) / y_std) * torch.exp(1j * self.y.angle())

    def scaling(self) -> None:
        """
        Perform scaling on the data.
        """
        x_std = self.x.std()
        y_std = self.y.std()
        if x_std != 0:
            self.x = self.x / x_std
        if y_std != 0:
            self.y = self.y / y_std

def get_loaders(
    train_path: str,
    val_path: str | dict[str],
    augmentation: str = None,
    transform: str = None,
    dim: int = 2,
    batch_size: int = 16,
    num_workers: int = 5,
    pin_memory: bool = True,
    log: Logger = None,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Get a dataloader for training and validation.

    Parameters:
        train_path (str): Path to the training data.
        val_path (str | dict[str]): Path or dictionary of paths to the validation data. Defaults to an empty dictionary.
        augmentation (str): Type of data augmentation. Defaults to None.
        transform (str): Type of data transformation. Defaults to None.
        dim (int): Dimensionality of the data. Defaults to 2.
        batch_size (int): Number of samples in each mini-batch. Defaults to 16.
        num_workers (int): Number of workers for data loading. Defaults to 5.
        pin_memory (bool): Whether to use pinned memory for faster data transfer. Defaults to True.
        log (Logger): Logger object for logging. Defaults to None.

    Returns:
        tuple[DataLoader, DataLoader, int]: Tuple containing training loader, validation loader, and total dataset size.
    """

    start = time.time()

    # Define dataset parameters
    dataset_params = {
        "augmentation": augmentation,
        "dim": dim,
        "transform": transform,
    }

    DatasetClass = CreateDataset

    # Create train and validation datasets
    train_ds = DatasetClass(**dataset_params, data_path=train_path)
    val_ds = DatasetClass(**dataset_params, data_path=val_path)

    # Create DataLoader instances
    loader_params = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "shuffle": True,
    }

    train_loader = DataLoader(train_ds, **loader_params)
    val_loader = DataLoader(val_ds, **loader_params)

    end = time.time()

    if log is not None:
        log.info(f"Data loading time: {(end - start):.3f}sec")
        log.info(
            f"Found {len(train_ds)} examples in the training-set; {len(val_ds)} examples in the validation-set ..."
        )

    return train_loader, val_loader, (len(train_ds) + len(val_ds))


def get_test_loader(
    test_path: str,
    transform: str = None,
    dim: int = 2,
    filenum: int = 8,
    num_workers: int = 5,
    pin_memory: bool = True,
) -> tuple[DataLoader, int]:
    """
    Get a data loader for testing.

    Parameters:
        test_path (str): Path to the test data.
        transform (str): Type of data transformation. Defaults to None.
        dim (int): Dimensionality of the data. Defaults to 2.
        filenum (int): Number of samples in each mini-batch. Defaults to 8.
        num_workers (int): Number of workers for data loading. Defaults to 5.
        pin_memory (bool): Whether to use pinned memory for faster data transfer. Defaults to True.

    Returns:
        tuple[DataLoader, int]: Tuple containing the test loader and the total number of examples in the dataset.
    """

    start = time.time()

    # Define dataset parameters
    dataset_params = {
        "dim": dim,
        "transform": transform,
    }

    test_ds = CreateDataset(**dataset_params, data_path=test_path)

    test_loader = DataLoader(
        test_ds,
        batch_size=filenum,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    end = time.time()

    print(
        f"\nData loading time: {(end - start):.3f}sec.\nFound {len(test_ds)} examples in the dataset."
    )

    return test_loader, len(test_ds)
