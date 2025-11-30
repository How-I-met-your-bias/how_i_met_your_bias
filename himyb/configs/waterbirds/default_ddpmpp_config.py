"""
Nathan Roos


Configuration for DDPM++ training on Waterbirds
"""

import ml_collections


def get_default_ddpmpp_config():
    """
    Configuration for DDPM++ training on Waterbirds
    """
    config = ml_collections.ConfigDict()
    config.img_resolution = 64
    config.in_channels = 3
    config.out_channels = 3
    config.label_dim = 3
    config.model_channels = 128
    config.channel_mult = [1,2,2,2]
    config.channel_mult_emb = 4
    config.num_blocks = 4
    config.attn_resolutions = [16]
    config.dropout = 0.1
    config.label_dropout = 0.15
    return config
