"""
Nathan Roos

"""

import ml_collections


def get_default_optimizer_config():
    """
    Returns a default optimizer configuration for experiment on BiasedMNIST.
    The values are those of Karras 2022
    """
    config = ml_collections.ConfigDict()
    config.optimizer = "adam"
    config.lr = 1e-4
    config.betas = (0.9, 0.999)
    config.eps = 1e-8
    config.use_scheduler = True
    config.lr_min = 0.
    config.lr_schedule = "cosine"
    return config
