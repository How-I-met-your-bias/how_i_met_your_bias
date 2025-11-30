"""
Waterbirds dataset implementation from DDB
"""

# https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz

import os
import tarfile
from typing import List, Tuple, Generator, Union, Dict
import logging
from collections import defaultdict

import requests
import tqdm
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd

from himyb.datasets.dataset_utils import create_balanced_dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Waterbirds(torch.utils.data.Dataset):
    """
    class labels : 0 or 1
    bias labels : -1 or 1 (1 for aligned, -1 for conflicting)
    """

    DOWNLOAD_URL = (
        "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
    )

    IMG_AVG = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        env: str,
        root: str = "./data",
        transform=None,
        metadata_filename: str = "metadata.csv",
        return_index: bool = False,
        class_label: int = None,
        bias_label_fmt: str = "0;1",
    ):
        """
        Args:
            env (str): Options: "train", "val", "test".
            root (str): Root directory for the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            metadata_filename (str): Name of the metadata file.
            return_index (bool): Whether to return the index of the sample.
            class_label (int, optional): Class label to filter samples by. If None, all samples are included.
            bias_label_fmt (str): Format of the bias label. Options: "0;1" or "-1;1". "0;1" for the actual  bias labels, "-1;1" for conflicting (-1) vs aligned (1).
        """
        assert env in ["train", "val", "test"], "env must be one of train, val, test"
        assert class_label in [0, 1, None], "class_label must be 0, 1 or None"
        assert bias_label_fmt in [
            "0;1",
            "-1;1",
        ], "bias_label_fmt must be '0;1' or '-1;1'"
        self.root = root
        self.env = env
        self.metadata_filename = metadata_filename
        self.return_index = return_index
        self.bias_label_fmt = bias_label_fmt
        self.num_classes = 2 if class_label is None else 1

        if not os.path.exists(os.path.join(self.root, "waterbirds")):
            self.__download_dataset()

        self.root = os.path.join(self.root, "waterbirds")
        self.transform = transform
        self.metadata_path = os.path.join(self.root, self.metadata_filename)

        metadata_csv = pd.read_csv(self.metadata_path)
        env_to_split = {"train": 0, "val": 1, "test": 2}
        metadata_csv = metadata_csv.query(f"split == {env_to_split[self.env]}")

        self.samples = {}  # dict mapping sample index to sample info
        self.files_count = 0  # nb of samples seen so far
        align_indices = {0: [], 1: []}
        conflict_indices = {0: [], 1: []}
        for i, (_, sample_info) in enumerate(metadata_csv.iterrows()):
            # Filter out the class label
            if class_label is not None and int(sample_info["y"]) != class_label:
                continue
            cl_label = int(sample_info["y"])
            self.samples[self.files_count] = {
                "image_path": os.path.join(self.root, sample_info["img_filename"]),
                "class_label": cl_label,
                "bias_label": (
                    (-1 if int(sample_info["y"]) != int(sample_info["place"]) else 1)
                    if self.bias_label_fmt == "-1;1"
                    else int(sample_info["place"])
                ),
                "all_attrs": list((str(e) for e in sample_info)),
            }

            # Add the sample to the appropriate list
            bias_aligned = (
                (self.samples[self.files_count]["bias_label"] == 1)
                if self.bias_label_fmt == "-1;1"
                else (self.samples[self.files_count]["bias_label"] == cl_label)
            )
            if bias_aligned:
                align_indices[cl_label].append(self.files_count)
            else:
                conflict_indices[cl_label].append(self.files_count)

            self.files_count += 1
        self.align_indices = {i: torch.tensor(align_indices[i]) for i in range(2)}
        self.conflict_indices = {i: torch.tensor(conflict_indices[i]) for i in range(2)}

    def __download_dataset(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        output_path = os.path.join(
            self.root, "waterbird_complete95_forest2water2.tar.gz"
        )
        print(
            f"=> Downloading {os.path.basename(self.root)} dataset from {self.DOWNLOAD_URL}"
        )

        try:
            response = requests.get(Waterbirds.DOWNLOAD_URL, stream=True)
            response.raise_for_status()

            with open(output_path, mode="wb") as write_stream, tqdm.tqdm(
                desc=output_path,
                total=int(response.headers["content-length"], 0),
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    write_stream.write(chunk)
                    pbar.update(len(chunk))

        except Exception as exc:
            raise RuntimeError(
                "Unable to complete dataset download, check for your internet connection or try changing download link."
            ) from exc

        print(
            f"=> Extracting waterbird_complete95_forest2water2.tar.gz to directory {self.root}"
        )
        try:
            with tarfile.open(output_path, mode="r:gz") as unballer:
                unballer.extractall(self.root)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to extract {output_path}, an error occured."
            ) from exc
        # Rename the extracted folder to "waterbirds"
        os.rename(
            os.path.join(self.root, "waterbird_complete95_forest2water2"),
            os.path.join(self.root, "waterbirds"),
        )
        os.remove(output_path)

    def __len__(self) -> int:
        return self.files_count

    def __getitem__(self, index: Union[int, slice, list]):

        if isinstance(index, slice):
            return [self.__getitem__(i) for i in range(*index.indices(len(self)))]

        if isinstance(index, list):
            return [self.__getitem__(idx) for idx in index]

        image = Image.open(self.samples[index]["image_path"])
        # image = (Image.open(self.samples[index]["image_path"]))
        image = self.transform(image)  # ["image"]
        class_label = self.samples[index]["class_label"]
        bias_label = self.samples[index]["bias_label"]

        data_dict = {
            "name": self.samples[index]["image_path"],
            "image": image,
            "class_label": class_label,
            "bias_label": bias_label,
        }

        if self.return_index:
            data_dict["index"] = index

        return data_dict

    def perclass_populations(
        self, return_labels: bool = False
    ) -> Union[Tuple[float, float], Tuple[Tuple[float, float], torch.Tensor]]:
        labels = torch.zeros(len(self))
        for i in range(self.files_count):
            labels[i] = self.samples[i]["class_label"]

        _, pop_counts = labels.unique(return_counts=True)

        if return_labels:
            return pop_counts.long(), labels.long()

        return pop_counts

    def get_bias_labels(self) -> Generator[None, None, torch.Tensor]:
        # left for eventual backward compatibility
        for i in range(len(self)):
            yield self.samples[i]["bias_label"]

    def get_class_and_bias_labels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the class and bias labels of all the samples of the dataset.
        """
        cl_labels = torch.zeros(len(self))
        bias_labels = torch.zeros(len(self))
        for i in range(len(self)):
            cl_labels[i] = self.samples[i]["class_label"]
            bias_labels[i] = self.samples[i]["bias_label"]
        return cl_labels.long(), bias_labels.long()

    def __repr__(self) -> str:
        return f"Waterbirds(env={self.env}, bias_amount=Fixed, num_classes={self.num_classes})"


class LargeWaterbirds(torch.utils.data.Dataset):
    """
    Combines train, val and test splits of the Waterbirds dataset.
    Support for aligned and conflicting samples indexing.
    """

    def __init__(
        self,
        root: str = "./data",
        transform=None,
        metadata_filename: str = "metadata.csv",
        return_index: bool = False,
        class_label: int = None,
        bias_label_fmt: str = "0;1",
    ):
        super().__init__()
        self.datasets = [
            Waterbirds(
                env=env,
                root=root,
                transform=transform,
                metadata_filename=metadata_filename,
                return_index=return_index,
                class_label=class_label,
                bias_label_fmt=bias_label_fmt,
            )
            for env in ["train", "val", "test"]
        ]

        self.len_train = len(self.datasets[0])
        self.len_val = len(self.datasets[1])
        self.len_test = len(self.datasets[2])
        self.align_indices = {
            i: torch.concat(
                (
                    self.datasets[0].align_indices[i],
                    self.datasets[1].align_indices[i] + self.len_train,
                    self.datasets[2].align_indices[i] + self.len_train + self.len_val,
                )
            )
            for i in range(2)
        }
        self.conflict_indices = {
            i: torch.concat(
                (
                    self.datasets[0].conflict_indices[i],
                    self.datasets[1].conflict_indices[i] + self.len_train,
                    self.datasets[2].conflict_indices[i]
                    + self.len_train
                    + self.len_val,
                )
            )
            for i in range(2)
        }
        self.files_count = self.len_train + self.len_val + self.len_test

    def __len__(self) -> int:
        return self.files_count

    def __getitem__(self, index: Union[int, slice, list]):
        if isinstance(index, slice):
            return [self.__getitem__(i) for i in range(*index.indices(len(self)))]

        if isinstance(index, list):
            return [self.__getitem__(idx) for idx in index]

        # image = self.transform(Image.open(self.samples[index]["image_path"]))
        index = int(index)
        if index < self.len_train:
            return self.datasets[0][index]
        elif index < self.len_train + self.len_val:
            return self.datasets[1][index - self.len_train]
        else:
            return self.datasets[2][index - self.len_train - self.len_val]


def get_balanced_waterbirds(
    root: str = "./data",
    transform=None,
    return_index: bool = False,
    bias_label_fmt: str = "0;1",
    rho: float = None,
    align_count: int = None,
    conflict_count: int = None,
) -> torch.utils.data.Dataset:
    """
    Get a version of the waterbirds dataset with images from all splits and a specified
    amount of aligned and conflicting samples per class.

    Args:
        root (str): Root directory for the dataset.
        transform (callable, optional): Optional transform to be applied on a sample.
        return_index (bool): Whether to return the index of the sample.
        bias_label_fmt (str): Format of the bias label. Options: "0;1" or "-1;1".
        rho (float, optional): Ratio of aligned samples to total samples. If None, align_count and conflict_count must be provided.
        align_count (int, optional): Number of aligned samples per class. If None, rho must be provided.
        conflict_count (int, optional): Number of conflicting samples per class. If None, rho must be provided.
    """
    dataset = LargeWaterbirds(
        root=root,
        transform=transform,
        return_index=return_index,
        bias_label_fmt=bias_label_fmt,
    )
    return create_balanced_dataset(
        dataset, rho=rho, align_count=align_count, conflict_count=conflict_count
    )


def get_balanced_waterbirds_dataloader(
    root: str,
    batch_size: int,
    img_size=(64, 64),
    rho: float = None,
    align_count: int = None,
    conflict_count: int = None,
    bias_label_fmt: str = "0;1",
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    transform=None,
) -> torch.utils.data.DataLoader:
    """
    Get a dataloader for the balanced waterbirds dataset.

    Args:
        batch_size (int): Batch size for the dataloader.
        num_workers (int): Number of workers for the dataloader.
        shuffle (bool): Whether to shuffle the dataset.
        **kwargs: Additional arguments for the dataset.

    Returns:
        DataLoader: Dataloader for the balanced waterbirds dataset.
    """
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=Waterbirds.IMG_AVG, std=Waterbirds.IMG_STD),
            ]
        )

    dataset = get_balanced_waterbirds(
        root=root,
        transform=transform,
        rho=rho,
        align_count=align_count,
        conflict_count=conflict_count,
        bias_label_fmt=bias_label_fmt,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
