"""
Nathan Roos
"""

import ml_collections


def get_default_dataset_config():
    config = ml_collections.ConfigDict()
    config.name = "bffhq"
    config.root = "../downloaded_assets/bffhq"
    config.rho = 0.99
    config.resolution = (64, 64)
    return config
