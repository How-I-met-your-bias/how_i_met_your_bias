"""
Nathan Roos


Configuration for DDPM++ training on CIFAR10 (supposed to work well with BiasedMNIST as well)
"""

import ml_collections


def get_default_ddpmpp_config():
    """
    Configuration for DDPM++ training on CIFAR10 (supposed to work well with BiasedMNIST as well)
    from "Elucidating the design space of diffusion-based generative models" Karras2022
    """
    config = ml_collections.ConfigDict()
    config.img_resolution = 32
    config.in_channels = 3
    config.out_channels = 3
    config.label_dim = 0
    config.model_channels = 128
    config.channel_mult = None
    config.channel_mult_emb = 4
    config.num_blocks = 4
    config.attn_resolutions = None
    config.dropout = 0.1
    config.label_dropout = 0.1
    return config
