"""
Nathan Roos

Provides unified function to retrieve all dataloaders
"""

import torch
import ml_collections

from himyb.datasets.biased_mnist import get_dataloader as bmnist_get_dataloader
from himyb.datasets.waterbirds import get_balanced_waterbirds_dataloader
from himyb.datasets.bffhq import get_balanced_bffhq_dataloader


def get_dataloader(
    batch_size: int,
    dataset_conf: ml_collections.ConfigDict,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Args:
        batch_size (int): The batch size to use
        dataset_conf (ml_collections.ConfigDict): The dataset configuration
    """
    if dataset_conf.name == "biased_mnist":
        dataloader = bmnist_get_dataloader(
            root=dataset_conf.root,
            train=dataset_conf.train,
            batch_size=batch_size,
            rho=dataset_conf.rho,
            n_confusing_labels=dataset_conf.n_confusing_labels,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            resolution=dataset_conf.resolution,
            resize_before_colouring=dataset_conf.resize_before_colouring,
            no_digit=dataset_conf.no_digit,
            classes_to_use=dataset_conf.classes_to_use,
            class_size=dataset_conf.class_size,
        )

    elif dataset_conf.name == "waterbirds":
        dataloader = get_balanced_waterbirds_dataloader(
            root=dataset_conf.root,
            batch_size=batch_size,
            img_size=dataset_conf.resolution,
            rho=dataset_conf.rho,
            align_count=dataset_conf.align_count,
            conflict_count=dataset_conf.conflict_count,
            bias_label_fmt=dataset_conf.bias_label_fmt,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
        )
    elif dataset_conf.name == "bffhq":
        dataloader = get_balanced_bffhq_dataloader(
            root=dataset_conf.root,
            batch_size=batch_size,
            img_size=dataset_conf.resolution,
            rho=dataset_conf.rho,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_conf.name}")
    return dataloader
