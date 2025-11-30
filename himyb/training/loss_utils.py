"""
Nathan Roos

"""

import torch


class EDMLoss:
    """
    Loss proposed in the paper "Elucidating the Design Space of Diffusion-Based Generative
    Models" (EDM).

    Warning : this loss takes a model that is parameterized to predict
    the fully denoised image y from the noised image y+n.
    """

    def __init__(self, p_mean=-1.2, p_std=1.2, sigma_data=0.5):
        """
        The noise is sampled following a log normal distribution such that :
        $\ln(\sigma) \sim \mathcal{N}(P_{mean}, P_{std}^2)$

        Args:
            p_mean(float): mean of the normal distribution (followed by ln(sigma))
            p_std(float): standard deviation of the normal distribution (followed by ln(sigma))
            sigma_data(float): noise level of the data ($=\sigma_{data}$ in the paper)
        """
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data

    def __call__(self, model, images, class_labels=None):
        """
        Args:
            model(torch.nn.Module): model to compute the loss (should be parameterized to predict \\
                the fully denoised image y from the noised image y+n)
            images(torch.Tensor): Shape (batch_size, C, H, W) batch of images
            labels(torch.Tensor): Shape (batch_size,) batch of class labels
        Returns:
            loss(torch.Tensor): Shape (batch_size, C, H, W) batch of losses WARNING THIS IS WEIRD
        """

        # noise level following the distribution $p_{train}$
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()

        # $=\lambda(\sigma)$ in the paper
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        y = images  # following the distribution $p_{data}$
        n = torch.randn_like(y) * sigma
        D_yn = model(y + n, sigma, class_labels)
        loss = (weight * (((D_yn - y) ** 2)).sum(dim=(1, 2, 3))).mean()
        return loss
