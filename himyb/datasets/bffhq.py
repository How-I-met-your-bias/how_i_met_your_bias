"""

The class BFFHQ is from the codebase 
https://anonymous.4open.science/r/DiffusingDeBias/DiffusingBias/Datasets/bffhq.py

The class LargeBFFHQ is new and wraps the train, val and test splits of the BFFHQ dataset
to create a single bigger dataset that enables rho selection where rho is the 
proportion of bias-aligned samples in the dataset.
"""

import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Callable, Tuple, Generator
import albumentations as A
import gdown
import zipfile

import himyb.datasets.dataset_utils as dataset_utils


def list_to_tensor_in_dict(dict: dict) -> dict:
    """
    Convert lists in a dictionary to numpy arrays.
    Args:
        dict (dict): Dictionary with lists as values.
    Returns:
        dict: Dictionary with numpy arrays as values.
    """
    for key, value in dict.items():
        if isinstance(value, list):
            dict[key] = torch.tensor(value)
    return dict


class BFFHQ(Dataset):
    DOWNLOAD_URL = "https://drive.google.com/file/d/1Y4y4vYz6sRJRqS9jJyD06cUSR618g0Rp/view?usp=sharing"
    DATASET_NAME = "bffhq"

    def __init__(
        self,
        root="./data/bffhq",
        env="train",
        transform=None,
        return_index=False,
        class_label: int = None,
    ):
        self.root = root
        self.transform = transform
        self.env = env
        self.return_index = return_index
        self.num_classes = 2 if class_label is None else 1

        if not os.path.isdir(os.path.join(self.root, BFFHQ.DATASET_NAME)):
            self.__download_dataset()
        else:
            self.data_dir = os.path.join(self.root, BFFHQ.DATASET_NAME)

        self.population_sizes = torch.zeros(2, 2)  # class; bias
        if self.env == "train":
            (
                self.samples,
                self.class_labels,
                self.bias_labels,
                self.align_indices,
                self.conflict_indices,
            ) = self.load_train_samples()

        if self.env == "val":
            (
                self.samples,
                self.class_labels,
                self.bias_labels,
                self.align_indices,
                self.conflict_indices,
            ) = self.load_val_samples()

        if self.env == "test":
            (
                self.samples,
                self.class_labels,
                self.bias_labels,
                self.align_indices,
                self.conflict_indices,
            ) = self.load_test_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        idx = int(idx)
        file_path = self.samples[idx]
        class_label = self.class_labels[idx]
        bias_label = self.bias_labels[idx]

        image = np.array(Image.open(file_path))
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        if self.return_index:
            return image, class_label, bias_label, idx

        data_dict = {
            "name": file_path,
            "image": image,
            "class_label": class_label,
            "bias_label": bias_label,
        }

        if self.return_index:
            data_dict["index"] = idx

        return data_dict

    def __download_dataset(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        output_path = os.path.join(self.root, "bffhq.zip")
        print(f"=> Downloading {BFFHQ.DATASET_NAME} for {BFFHQ.DOWNLOAD_URL}")

        try:
            gdown.download(id="1Y4y4vYz6sRJRqS9jJyD06cUSR618g0Rp", output=output_path)
        except:
            raise RuntimeError(
                "Unable to complete dataset download, check for your internet connection or try changing download link."
            )

        print(f"=> Extracting bffhq.zip to directory {self.root}")
        try:
            with zipfile.ZipFile(output_path, mode="r") as unzipper:
                unzipper.extractall(self.root)
        except:
            raise RuntimeError(f"Unable to extract {output_path}, an error occured.")

        self.data_dir = os.path.join(self.root, BFFHQ.DATASET_NAME)
        os.remove(output_path)

    def load_train_samples(self, class_label=None):
        """
        Load training samples to memory.
        Args:
            class_label (int, optional): Class label to filter samples. Defaults to None.
        """
        samples_path: List[str] = []
        class_labels: List[int] = []
        bias_labels: List[int] = []
        align_indices = {0: [], 1: []}
        conflict_indices = {0: [], 1: []}
        cur_sample_idx = 0

        bias_folder = "0.5pct"

        for class_folder in sorted(
            os.listdir(os.path.join(self.data_dir, bias_folder, "align"))
        ):
            for filename in sorted(
                os.listdir(
                    os.path.join(self.data_dir, bias_folder, "align", class_folder)
                )
            ):

                if class_label is not None and class_label != self.assign_class_label(
                    filename
                ):
                    continue

                samples_path.append(
                    os.path.join(
                        self.data_dir, bias_folder, "align", class_folder, filename
                    )
                )
                sample_class_label = self.assign_class_label(filename)
                sample_bias_label = self.assign_bias_label(filename)
                class_labels.append(sample_class_label)
                bias_labels.append(sample_bias_label)
                align_indices[sample_class_label].append(cur_sample_idx)
                cur_sample_idx += 1
                self.population_sizes[sample_class_label, sample_bias_label] += 1

        for class_folder in sorted(
            os.listdir(os.path.join(self.data_dir, bias_folder, "conflict"))
        ):
            if class_label is not None and class_label != self.assign_class_label(
                filename
            ):
                continue

            for filename in sorted(
                os.listdir(
                    os.path.join(self.data_dir, bias_folder, "conflict", class_folder)
                )
            ):
                if class_label is not None and class_label != self.assign_class_label(
                    filename
                ):
                    continue
                samples_path.append(
                    os.path.join(
                        self.data_dir,
                        bias_folder,
                        "conflict",
                        class_folder,
                        filename,
                    )
                )
                sample_class_label = self.assign_class_label(filename)
                sample_bias_label = self.assign_bias_label(filename)
                class_labels.append(sample_class_label)
                bias_labels.append(sample_bias_label)
                conflict_indices[sample_class_label].append(cur_sample_idx)
                cur_sample_idx += 1
                self.population_sizes[sample_class_label, sample_bias_label] += 1

        return (
            np.array(samples_path),
            np.array(class_labels),
            np.array(bias_labels),
            list_to_tensor_in_dict(align_indices),
            list_to_tensor_in_dict(conflict_indices),
        )

    def load_val_samples(self, class_label=None):
        """
        Load validation samples to memory.
        Args:
            class_label (int, optional): Class label to filter samples. Defaults to None.
        """
        samples_path: List[str] = []
        class_labels: List[int] = []
        bias_labels: List[int] = []
        align_indices = {0: [], 1: []}
        conflict_indices = {0: [], 1: []}
        cur_sample_idx = 0

        for filename in sorted(os.listdir(os.path.join(self.data_dir, "valid"))):
            if class_label is not None and class_label != self.assign_class_label(
                filename
            ):
                continue

            samples_path.append(os.path.join(self.data_dir, "valid", filename))
            sample_class_label = self.assign_class_label(filename)
            sample_bias_label = self.assign_bias_label(filename)
            class_labels.append(sample_class_label)
            bias_labels.append(sample_bias_label)
            align_indices[sample_class_label].append(cur_sample_idx)
            cur_sample_idx += 1
            self.population_sizes[sample_class_label, sample_bias_label] += 1

        return (
            np.array(samples_path),
            np.array(class_labels),
            np.array(bias_labels),
            list_to_tensor_in_dict(align_indices),
            list_to_tensor_in_dict(conflict_indices),
        )

    def load_test_samples(self, class_label=None):
        """
        Load test samples to memory.
        Args:
            class_label (int, optional): Class label to filter samples. Defaults to None.
        """
        samples_path: List[str] = []
        class_labels: List[int] = []
        bias_labels: List[int] = []
        align_indices = {0: [], 1: []}
        conflict_indices = {0: [], 1: []}
        cur_sample_idx = 0

        for filename in sorted(os.listdir(os.path.join(self.data_dir, "test"))):

            if class_label is not None and class_label != self.assign_class_label(
                filename
            ):
                continue

            samples_path.append(os.path.join(self.data_dir, "test", filename))
            sample_class_label = self.assign_class_label(filename)
            sample_bias_label = self.assign_bias_label(filename)
            class_labels.append(sample_class_label)
            bias_labels.append(sample_bias_label)
            if sample_class_label == sample_bias_label:
                align_indices[sample_class_label].append(cur_sample_idx)
            else:
                conflict_indices[sample_class_label].append(cur_sample_idx)
            cur_sample_idx += 1
            self.population_sizes[sample_class_label, sample_bias_label] += 1

        return (
            np.array(samples_path),
            np.array(class_labels),
            np.array(bias_labels),
            list_to_tensor_in_dict(align_indices),
            list_to_tensor_in_dict(conflict_indices),
        )

    def assign_bias_label(self, filename: str) -> int:
        no_extension = filename.split(".")[0]
        _, y, z = no_extension.split("_")
        y = int(y)
        z = int(z)
        return z

    def assign_class_label(self, filename: str):
        no_extension = filename.split(".")[0]
        _, y, _ = no_extension.split("_")
        return int(y)

    def get_bias_labels(self) -> Generator[None, None, torch.Tensor]:
        for i in range(len(self)):
            yield self[i]["bias_label"]

    def __repr__(self) -> str:
        return f"BFFHQ(env={self.env}, num_classes={self.num_classes})"


class LargeBFFHQ(Dataset):
    """
    Combines the train, test and val splits of the BFFHQ dataset into a single dataset
    in order to choose the rho level
    """

    def __init__(self, root="./data/bffhq", transform=None, class_label=None):
        self.root = root
        self.transform = transform
        self.num_classes = 2 if class_label is None else 1
        self.class_label = class_label

        self.datasets = [
            BFFHQ(root=self.root, env=env, transform=self.transform)
            for env in ["train", "val", "test"]
        ]

        self.len_train = len(self.datasets[0])
        self.len_val = len(self.datasets[1])
        self.len_test = len(self.datasets[2])
        self.len = self.len_train + self.len_val + self.len_test

        self.align_indices = self.align_indices = {
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

    def __getitem__(self, idx):
        if idx < self.len_train:
            return self.datasets[0][idx]
        elif idx < self.len_train + self.len_val:
            return self.datasets[1][idx - self.len_train]
        else:
            return self.datasets[2][idx - self.len_train - self.len_val]

    def __len__(self):
        return self.len


def get_balanced_bffhq(
    root: str,
    rho: float,
    transform: Callable = None,
):
    dataset = LargeBFFHQ(root=root, transform=transform)
    return dataset_utils.create_balanced_dataset(dataset, rho)


def get_balanced_bffhq_dataloader(
    root: str,
    rho: float,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    transform: Callable = None,
    img_size: Tuple[int, int] = (64, 64),
):
    if transform is None:
        transform = A.Compose(
            [
                A.Resize(img_size[0], img_size[1]),
                A.Normalize(normalization="standard"),
                A.pytorch.ToTensorV2(),
            ]
        )

    dataset = get_balanced_bffhq(root, rho, transform)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )
