"""
Nathan Roos

"""

import torch


def get_optimizer(optimizer_config, model):
    """
    Args:
        optimizer_config (ml_collections.ConfigDict): configuration for the optimizer
        model (torch.nn.Module): model to optimize
    Returns:
        (torch.optim.Optimizer)
    """
    conf = optimizer_config
    if conf.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=conf.lr,
            betas=conf.betas,
            eps=conf.eps,
        )
    else:
        raise ValueError(f"Unknown optimizer {optimizer_config.optimizer}")
    return optimizer
