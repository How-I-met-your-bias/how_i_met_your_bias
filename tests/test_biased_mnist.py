"""
Nathan Roos
"""

import os
import sys
import shutil

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import himyb.datasets.biased_mnist as biased_mnist


@pytest.fixture
def root_dir():
    """Return the directory where the Biased MNIST dataset is stored"""
    return "../biits/downloaded_assets/biased_mnist"


def test_biased_mnist_dataloader():
    """Test the creation of the dataloader"""
    batch_size = 100
    resolution = (32, 32)
    tmp_dir = "./tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    custom_loader = biased_mnist.get_dataloader(
        root=tmp_dir,
        batch_size=100,
        train=True,
        rho=0.9,
        n_confusing_labels=9,
        resolution=resolution,
    )
    assert custom_loader is not None

    images, class_labels, bias_labels = next(iter(custom_loader))
    assert images.shape == (batch_size, 3, *resolution)
    assert class_labels.shape == (batch_size,)
    assert bias_labels.shape == (batch_size,)
    shutil.rmtree(tmp_dir)
