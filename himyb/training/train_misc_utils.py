"""
Nathan Roos
"""

import random
from typing import Tuple, Union

import numpy as np
import torch


def get_num_params(model):
    """
    Get the number of parameters of a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seeds(seed):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The random seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt

        # define the attributes
        self.val = 0
        self.avg = 0
        self.std = 0
        self.sum = 0
        self.sq_sum = 0
        self.count = 0

    def reset(self):
        """Set all values to 0"""
        self.val = 0
        self.avg = 0
        self.std = 0
        self.sum = 0
        self.sq_sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update the values by adding n times the value val to the average"""
        assert (
            n >= 0
        ), "The number of samples to add to the average must be non-negative"
        self.val = val
        self.sum += val * n
        self.sq_sum += n * val**2
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
            self.std = (self.sq_sum / self.count - self.avg**2) ** 0.5

    def __str__(self):
        fmtstr = f"{self.name} {self.count} {self.avg} +/- {self.std}"
        return fmtstr


def prepare_batch(
    batch,
    device,
    dataset_name: str,
    ret_img: bool = True,
    ret_class_label: bool = True,
    ret_bias_label: bool = False,
)-> Union[torch.Tensor, Tuple[torch.Tensor]]:
    """
    Extract specific elements from the batch
    Adapt to the different datasets
    Args:
        batch: The batch to extract from
        device: The device to move the tensors to
        dataset_name: The name of the dataset
        ret_img: Whether to return the image
        ret_class_label: Whether to return the class label
        ret_bias_label: Whether to return the bias label
    """
    if dataset_name == "biased_mnist":
        image, class_labels, bias_labels = batch
    elif dataset_name == "waterbirds":
        image = batch["image"]
        class_labels = batch["class_label"]
        bias_labels = batch["bias_label"]
    elif dataset_name == "bffhq":
        image = batch["image"]
        class_labels = batch["class_label"]
        bias_labels = batch["bias_label"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    outputs = []
    if ret_img:
        outputs.append(image.to(device))
    if ret_class_label:
        #add 1 to the class labels to make them start at 1 instead of 0 (0 is uncond label)
        outputs.append(class_labels.to(device)+1)
    if ret_bias_label:
        outputs.append(bias_labels.to(device))
    return tuple(outputs)
