"""
Nathan Roos
"""

import ml_collections


def get_default_dataset_config():
    config = ml_collections.ConfigDict()
    config.name = "biased_mnist"
    config.root = "../downloaded_assets/biased_mnist"
    config.rho = 0.99
    config.train = True
    config.n_confusing_labels = 9
    config.classes_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    config.resolution = (32, 32)
    config.resize_before_colouring = False
    config.no_digit = False
    config.class_size = None
    return config
