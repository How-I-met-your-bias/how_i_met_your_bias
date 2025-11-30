"""
Nathan Roos

Based on :  ReBias
            Copyright (c) 2020-present NAVER Corp.
            MIT license
            Python implementation of Biased-MNIST.
"""

import os
import random
import numpy as np
from PIL import Image

import torch

from torchvision import transforms
from torchvision.datasets import MNIST


class BiasedMNIST(MNIST):
    """
    Compared to the original implementation, we add the following features:
    * possibility to keep only a subset of the classes by setting `classes_to_use` to a list of integers,
    * possibility to keep only a subset of the samples per class by setting `class_size`
    to an integer (if not None, only class_size samples per class are kept).


    ORIGINAL DOCUMENTATION:
    A base class for Biased-MNIST.
    We manually select ten colours to synthetic colour bias.
    (See `COLOUR_MAP` for the colour configuration)
    Usage is exactly same as torchvision MNIST dataset class.

    You have two parameters to control the level of bias.

    Parameters
    ----------
    root : str
        path to MNIST dataset.
    rho : float, default=1.0
        Here, each class has the pre-defined colour (bias).
        data_label_correlation, or `rho` controls the level of the dataset bias.
        rho can be a float, in which case all classes have the same biase level,
        or it can be a list of floats of size nb_class,
        in which case each class has a different bias level.

        A sample is coloured with
            - the pre-defined colour with probability `rho`,
            - coloured with one of the other colours with probability `1 - rho`.
              The number of ``other colours'' is controlled by `n_confusing_labels` (default: 9).
        Note that the colour is injected into the background of the image (see `_binary_to_colour`).

        Hence, we have
            - Perfectly biased dataset with rho=1.0
            - Perfectly unbiased with rho=0.1 (1/10) ==> our ``unbiased'' setting in the test time.

    n_confusing_labels : int, default=9
        In the real-world cases, biases are not equally distributed, but highly unbalanced.
        We mimic the unbalanced biases by changing the number of confusing colours for each class.
        In the paper, we use n_confusing_labels=9, i.e., during training, the model can observe
        all colours for each class.
        However, you can make the problem harder by setting smaller n_confusing_labels, e.g., 2.
        We suggest to researchers considering this benchmark for future researches.
    """

    # list of RGB colours to use for the bias of each digit
    COLOUR_MAP = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [225, 225, 0],
        [225, 0, 225],
        [0, 255, 255],
        [255, 128, 0],
        [255, 0, 128],
        [128, 0, 255],
        [128, 128, 128],
    ]

    COLOUR_NAMES = [
        "red",
        "green",
        "blue",
        "yellow",
        "magenta",
        "cyan",
        "orange",
        "rose",
        "electric violet",
        "grey",
    ]

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        rho=1.0,
        n_confusing_labels=9,
        classes_to_use=None,
        class_size=None,  # if not None, only keep class_size samples per class
    ):
        assert classes_to_use is None or all(
            [cl in range(10) for cl in classes_to_use]
        ), "classes_to_use should be a list of integers between 0 and 9"
        self.classes_to_use = (
            list(range(10)) if classes_to_use is None else classes_to_use
        )
        self.num_classes = len(self.classes_to_use)
        self.n_confusing_labels = n_confusing_labels
        assert (
            self.num_classes > self.n_confusing_labels and self.n_confusing_labels > 0
        )

        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.random = True
        self.class_size = class_size

        if isinstance(rho, float):
            self.rho = [rho] * 10
        else:
            if len(rho) != 10:
                raise ValueError(
                    "rho should be a float or a list of 10 floats,"
                    + f"but it is a {type(rho)}"
                    + f"of length {len(rho)}"
                )
            else:
                self.rho = rho

        self.data, self.targets, self.biased_targets = self.build_biased_mnist()

        indices = np.arange(len(self.data))
        self._shuffle(indices)

        self.data = self.data[indices].numpy()
        self.targets = self.targets[indices]
        self.biased_targets = self.biased_targets[indices]

        self.align_count = self.biased_targets.eq(self.targets).sum().item()
        self.conflict_count = len(self) - self.align_count
        self.effective_rho = (self.biased_targets == self.targets).float().mean().item()

    @property
    def raw_folder(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, "processed")

    def _shuffle(self, iterable):
        if self.random:
            np.random.shuffle(iterable)

    def _make_biased_mnist(self, indices, label):
        raise NotImplementedError

    def _update_bias_indices(self, bias_indices, label: int):

        # indices of the samples having class "label"
        indices = np.where((self.targets == label).numpy())[0]
        self._shuffle(indices)
        indices = torch.LongTensor(indices)
        if self.class_size is not None:
            indices = indices[: self.class_size]

        n_samples = len(indices)
        n_correlated_samples = int(n_samples * self.rho[label])
        n_decorrelated_per_class = int(
            np.ceil((n_samples - n_correlated_samples) / (self.n_confusing_labels))
        )

        correlated_indices = indices[:n_correlated_samples]
        bias_indices[label] = torch.cat([bias_indices[label], correlated_indices])

        decorrelated_indices = torch.split(
            indices[n_correlated_samples:], n_decorrelated_per_class
        )

        other_labels = random.sample(
            [_label for _label in self.classes_to_use if _label != label],
            self.n_confusing_labels,
        )
        self._shuffle(other_labels)

        for idx, _indices in enumerate(decorrelated_indices):
            _label = other_labels[idx]
            bias_indices[_label] = torch.cat([bias_indices[_label], _indices])

    def build_biased_mnist(self):
        """Build biased MNIST."""

        # indices of the samples having bias class "label"
        bias_indices = {label: torch.LongTensor() for label in self.classes_to_use}
        for label in self.classes_to_use:
            self._update_bias_indices(bias_indices, label)

        data = torch.ByteTensor()
        targets = torch.LongTensor()
        biased_targets = []

        for bias_label, indices in bias_indices.items():
            _data, _targets = self._make_biased_mnist(indices, bias_label)
            data = torch.cat([data, _data])
            targets = torch.cat([targets, _targets])
            biased_targets.extend([bias_label] * len(indices))

        biased_targets = torch.LongTensor(biased_targets)
        return data, targets, biased_targets

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.astype(np.uint8), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, int(self.biased_targets[index])


class ColourBiasedMNIST(BiasedMNIST):
    """
    Compared to the original implementation, we add the following features:
    * possibility to change the resolution of the images. We resize the images \
    to the desired resolution BEFORE colouring them (in case you want to resize them AFTER, \
    you can just add a resizing operation to the transform),
    * possibility to keep only the background (no digit) by setting `no_digit=True`.
    * possibility to keep only a subset of the classes by setting `classes_to_use` to a list of integers,
    * possibility to keep only a subset of the samples per class by setting `class_size`
    to an integer (if not None, only class_size samples per class are kept).
    """

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        rho=1.0,
        resolution=(28, 28),
        n_confusing_labels=9,
        no_digit=False,  # if True : only keep background
        classes_to_use=None,  # if not None, only keep these classes
        class_size=None,  # if not None, only keep class_size samples per class
    ):
        # these lines must be executed before the super().__init__ call
        self.do_resize = resolution != (28, 28)
        if self.do_resize:
            self.transform_resize = transforms.Resize(resolution)
        self.no_digit = no_digit
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            rho=rho,
            n_confusing_labels=n_confusing_labels,
            classes_to_use=classes_to_use,
            class_size=class_size,
        )

    def _binary_to_colour(self, data, colour):
        """
        Colour the data with the given colour.

        Args:
            data (torch.Tensor): (num_images, height, width) grey scale image
            colour (list): RGB colour
        """
        if self.no_digit:
            data = torch.zeros_like(data)
        assert len(colour) == 3
        assert data.dim() == 3
        data = torch.stack([data, data, data], dim=1)
        if self.do_resize:
            data = self.transform_resize(data)
        fg_data = torch.zeros_like(data)
        fg_data[data != 0] = 255
        fg_data[data == 0] = 0

        bg_data = torch.zeros_like(data)
        bg_data[data == 0] = 1
        bg_data[data != 0] = 0
        bg_data = (bg_data.permute(0, 2, 3, 1) * torch.ByteTensor(colour)).permute(
            0, 3, 1, 2
        )

        data = fg_data + bg_data
        return data.permute(0, 2, 3, 1)

    def _make_biased_mnist(self, indices, label):
        return (
            self._binary_to_colour(self.data[indices], self.COLOUR_MAP[label]),
            self.targets[indices],
        )


def get_dataloader(
    root,
    train=True,
    batch_size=128,
    shuffle=True,
    rho=0.9,
    n_confusing_labels=9,
    num_workers=4,
    pin_memory=True,
    resolution=(28, 28),
    resize_before_colouring=False,
    no_digit=False,
    classes_to_use=None,
    class_size=None,
):
    """
    Return a data loader for the BiasedMNIST dataset (digits in white and background colored).
    The images are normalized to [-1, 1]. Be sure to renormalize them before plotting.

    Args:
        root (str): path to the MNIST dataset
        train (bool): whether to use the training set
        batch_size (int): batch size
        rho (float): correlation between the data and the label
        n_confusing_labels (int): number of confusing labels
        num_workers (int): number of workers for the data loader
        pin_memory (bool): whether to pin the memory
        resolution (tuple): resolution of the images (default: (28, 28))
        resize_before_colouring (bool): whether to resize the images before colouring them (False gives better defined edges)
        no_digit (bool): if True, only keep the colored background (the digit is not showed)
        classes_to_use (list): if not None, only keep these classes
        class_size (int): if not None, only keep class_size samples per class
    """
    if resize_before_colouring:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(resolution, antialias=True),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    dataset = ColourBiasedMNIST(
        root,
        train=train,
        download=True,
        transform=transform,
        rho=rho,
        n_confusing_labels=n_confusing_labels,
        resolution=resolution if resize_before_colouring else (28, 28),
        no_digit=no_digit,
        classes_to_use=classes_to_use,
        class_size=class_size,
    )

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
