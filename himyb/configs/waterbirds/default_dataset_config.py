"""
Nathan Roos
"""

import ml_collections


def get_default_dataset_config():
    config = ml_collections.ConfigDict()
    config.name = "waterbirds"
    config.root = "../downloaded_assets/waterbirds"
    config.bias_label_fmt = "0;1"
    config.rho = 0.95
    config.align_count = None
    config.conflict_count = None
    config.resolution = (64, 64)
    return config
