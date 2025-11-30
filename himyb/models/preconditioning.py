"""
Implements wrapper classes that uses diffusion models to parameterize the diffusion process

Modified by Nathan Roos from the implementation of Karras2022 :

    Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    This work is licensed under a Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License.
    You should have received a copy of the license along with this
    work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import torch


class EDMPrecond(torch.nn.Module):
    """
    Preconditioning from the paper "Elucidating the Design Space of Diffusion-Based Generative
      Models" (EDM).

    Given a model F(x, noise, class_labels),we parameterize the model : (noise is also called sigma)
    D(x, noise, class_labels) = c_skip * x + \\
        c_out(noise) * F(c_in(noise) * x, c_noise(noise), class_labels)
    """

    def __init__(
        self,
        model,
        sigma_data=0.5,  # Expected standard deviation of the training data.
    ):
        """
        Args:
            model(torch.nn.Module): Model to precondition (already initialized)
            sigma_data(float): Expected standard deviation of the training data.
        """
        super().__init__()
        self.sigma_data = sigma_data
        self.sigma_data_squared = sigma_data**2
        self.wrapped_model = model
        self.label_dim = self.wrapped_model.label_dim
        self.img_resolution = self.wrapped_model.img_resolution
        self.in_channels = self.wrapped_model.in_channels
        self.out_channels = self.wrapped_model.out_channels

    def forward(self, x, sigma, class_labels=None):
        """
        Class

        Args:
            x(torch.Tensor): Shape (batch_size, C, H, W) batch of images
            sigma(torch.Tensor): Shape (batch_size) Noise level
            class_labels(torch.Tensor): Shape (batch_size, ) or (batch_size, label_dim) Class labels
        """
        batch_size = x.shape[0]

        # handling of the two formats of class_labels accepted
        assert class_labels.shape in ((batch_size, self.label_dim), (batch_size,))
        # one hot encode the class labels if necessary
        if class_labels.shape == (batch_size,):
            class_labels = torch.nn.functional.one_hot(
                class_labels.to(torch.long), self.label_dim
            )
        class_labels = class_labels.float()

        sigma_data2_pl_sigma2 = self.sigma_data_squared + sigma**2
        c_skip = (self.sigma_data_squared / sigma_data2_pl_sigma2).reshape(-1, 1, 1, 1)
        c_in = (1 / torch.sqrt(sigma_data2_pl_sigma2)).reshape(-1, 1, 1, 1)
        c_out = (self.sigma_data * sigma / torch.sqrt(sigma_data2_pl_sigma2)).reshape(
            -1, 1, 1, 1
        )
        c_noise = (torch.log(sigma) / 4).reshape(-1, 1, 1, 1)
        f_x = self.wrapped_model(
            x=(x * c_in),
            noise_labels=c_noise.flatten(),
            class_labels=class_labels.clone(),
        )

        return c_skip * x + c_out * f_x
