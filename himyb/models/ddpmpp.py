"""
Modified by Nathan Roos

Implementation of the DDPM++ model from Karras 2022 and Song 2021
for the purpose of generating images of resolution 32*32*3

Previous work (https://github.com/NVlabs/edm) copyright mention :
    Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    This work is licensed under a Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License.
    You should have received a copy of the license along with this
    work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import torch
import torch.nn as nn
import numpy as np
import himyb.models.layers as layers


class DDPMPP(torch.nn.Module):
    """
    Implementation of the DDPM++ model from Karras 2022 and Song 2021
    for the purpose of generating images of resolution 32*32*3
    """

    def __init__(
        self,
        img_resolution=32,
        in_channels=3,
        out_channels=3,
        label_dim=0,
        model_channels=128,
        channel_mult=None,
        channel_mult_emb=4,
        num_blocks=4,
        attn_resolutions=None,
        dropout=0.1,
        label_dropout=0.1,
        verbose_init=False,
    ):
        """
        Args:
            img_resolution (int): resolution of the input image
            in_channels (int): number of input color channels
            out_channels (int): number of output color channels
            label_dim (int): 0 for unconditional model, num_classes+1 for conditional model
            model_channels=128 (int) : base multiplier for the number of channels in the model
            channel_mult (list[int]): list of channel multipliers per resolution
            channel_mult_emb (int): channel multiplier for the embedding vector
            num_blocks (int): number of residual block per resolution
            attn_resolutions (list[int]): list of resolutions for which to use self-attention (default: [16])
            dropout (float): dropout probability of intermediate activation
            label_dropout (float): dropout probability of the class label (=p_uncond) for CFG
            verbose_init (bool): print blocks and their channels during initialization
        """
        super().__init__()
        assert (
            img_resolution > 0
            and in_channels > 0
            and out_channels > 0
            and label_dim >= 0
        )
        assert channel_mult is None or all([c > 0 for c in channel_mult])
        assert channel_mult_emb > 0 and num_blocks > 0
        assert attn_resolutions is None or all([res > 0 for res in attn_resolutions])
        assert 0 <= dropout < 1 and 0 <= label_dropout < 1
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.noise_channels = model_channels
        self.channel_mult = channel_mult if channel_mult is not None else [2, 2, 2]
        self.bottom_level_idx = len(self.channel_mult) - 1
        self.channel_mult_emb = channel_mult_emb
        self.emb_channels = self.model_channels * self.channel_mult_emb
        self.num_blocks = num_blocks
        self.label_dim = label_dim
        self.is_conditional = label_dim > 0
        self.attn_resolutions = (
            attn_resolutions if attn_resolutions is not None else [16]
        )
        self.dropout = dropout
        self.label_dropout = label_dropout
        self.resample_filter = [1, 1]

        def print_verbose(*args, **kwargs):
            if verbose_init:
                print(*args, **kwargs)

        # initialization parameters of the layers
        init_default = dict(init_mode="xavier_uniform")  # default weight initialization
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)  # bias init
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(  # key word arguments common to all UNetBlocks
            emb_channels=self.emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=self.resample_filter,
            resample_proj=True,
            init=init_default,
            init_zero=init_zero,
            init_attn=init_attn,
        )

        # mapping from noise level (or timestep) to noise embedding
        self.map_noise_emb = layers.PositionalEmbedding(self.noise_channels)
        if self.is_conditional:
            # mapping from class labels to class embedding
            print_verbose(f"map_label {label_dim} -> {self.noise_channels}")
            self.map_label = layers.Linear(
                in_features=label_dim, out_features=self.noise_channels, **init_default
            )
        print_verbose(f"map_layer0 {self.noise_channels} -> {self.emb_channels}")
        self.map_layer0 = layers.Linear(
            in_features=self.noise_channels,
            out_features=self.emb_channels,
            **init_default,
        )
        print_verbose(f"map_layer1 {self.emb_channels} -> {self.emb_channels}")
        self.map_layer1 = layers.Linear(
            in_features=self.emb_channels,
            out_features=self.emb_channels,
            **init_default,
        )

        # construction encoder layers
        self.encoder = nn.ModuleDict()
        ch_in = in_channels  # temporary variable for the number of input channels
        ch_out = None  # temporary variable for the number of output channels
        for level, mult in enumerate(self.channel_mult):
            # current level resolution = img_resolution / 2^level
            res = img_resolution >> level
            if level == 0:
                ch_out = model_channels
                print_verbose(f"{res}x{res}_conv {ch_in} -> {ch_out}")
                self.encoder[f"{res}x{res}_conv"] = layers.Conv2d(
                    in_channels=ch_in, out_channels=ch_out, kernel=3, **init_default
                )
            else:
                print_verbose(f"{res}x{res}_down {ch_in} -> {ch_out}")
                self.encoder[f"{res}x{res}_down"] = layers.UNetBlock(
                    in_channels=ch_out, out_channels=ch_out, down=True, **block_kwargs
                )

            for res_block_idx in range(num_blocks):
                ch_in = ch_out
                ch_out = model_channels * mult
                use_attn = res in self.attn_resolutions
                print_verbose(f"{res}x{res}_block{res_block_idx} {ch_in} -> {ch_out}")
                self.encoder[f"{res}x{res}_block{res_block_idx}"] = layers.UNetBlock(
                    in_channels=ch_in,
                    out_channels=ch_out,
                    attention=use_attn,
                    **block_kwargs,
                )

        skips = [block.out_channels for _, block in self.encoder.items()]

        # construction decoder layers
        self.decoder = nn.ModuleDict()
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            # current level resolution = img_resolution / 2^level
            res = img_resolution >> level

            if level == self.bottom_level_idx:
                print_verbose(f"{res}x{res}_in0 {ch_out} -> {ch_out}")
                self.decoder[f"{res}x{res}_in0"] = layers.UNetBlock(
                    in_channels=ch_out,
                    out_channels=ch_out,
                    attention=True,
                    **block_kwargs,
                )
                print_verbose(f"{res}x{res}_in1 {ch_out} -> {ch_out}")
                self.decoder[f"{res}x{res}_in1"] = layers.UNetBlock(
                    in_channels=ch_out,
                    out_channels=ch_out,
                    attention=False,
                    **block_kwargs,
                )
            else:
                print_verbose(f"{res}x{res}_up {ch_out} -> {ch_out}")
                self.decoder[f"{res}x{res}_up"] = layers.UNetBlock(
                    in_channels=ch_out, out_channels=ch_out, up=True, **block_kwargs
                )

            for res_block_idx in range(num_blocks + 1):
                ch_in = ch_out + skips.pop()
                ch_out = model_channels * mult
                # use attention in the last block of the chosen resolutions
                use_attn = res_block_idx == num_blocks and res in self.attn_resolutions
                print_verbose(f"{res}x{res}_block{res_block_idx} {ch_in} -> {ch_out}")
                self.decoder[f"{res}x{res}_block{res_block_idx}"] = layers.UNetBlock(
                    in_channels=ch_in,
                    out_channels=ch_out,
                    attention=use_attn,
                    **block_kwargs,
                )
        print_verbose(f"{res}x{res}_aux_norm {ch_out}")
        self.decoder[f"{res}x{res}_aux_norm"] = layers.GroupNorm(
            num_channels=ch_out, eps=1e-6
        )
        print_verbose(f"{res}x{res}_aux_conv {ch_out} -> {out_channels}")
        self.decoder[f"{res}x{res}_aux_conv"] = layers.Conv2d(
            in_channels=ch_out, out_channels=self.out_channels, kernel=3, **init_zero
        )

    def get_embedding(self, noise_labels, class_labels=None):
        """
        Compute the aggregate of noise and class embeddings

        Args:
            noise_labels (torch.Tensor): Shape (batch_size,) noise (timestep) labels
            class_labels (torch.Tensor): Shape (batch_size, num_classes+1) class labels \
                one-hot vector. Class 0 is the unconditional class.
        """
        # Assert explanation :
        # first condition: if the model is conditional, the class labels must have the right shape
        # second condition: if the model is unconditional, there should be no class labels
        assert (
            self.is_conditional
            and class_labels is not None
            and class_labels.shape[1] == self.label_dim
        ) or (not self.is_conditional and class_labels is None)
        batch_size = noise_labels.shape[0]

        noise_emb = self.map_noise_emb(noise_labels)
        # swap sin/cos as in the original implementation
        noise_emb = (
            noise_emb.reshape(batch_size, 2, -1).flip(1).reshape(*noise_emb.shape)
        )
        complete_emb = noise_emb
        if self.is_conditional:
            # drop a proportion label_dropout of the class labels (change the class label to 0)
            if self.training and self.label_dropout:
                class_labels = class_labels * (
                    torch.rand([batch_size, 1], device=class_labels.device)
                    >= self.label_dropout
                ).to(class_labels.dtype)
            # scale the class labels by sqrt of the number of classes
            class_labels *= np.sqrt(self.map_label.in_features)
            complete_emb += self.map_label(class_labels)
        complete_emb = nn.functional.silu(self.map_layer0(complete_emb))
        complete_emb = nn.functional.silu(self.map_layer1(complete_emb))

        return complete_emb

    def forward(self, x, noise_labels, class_labels):
        """
        Args
            x (torch.Tensor): Shape (batch_size, in_channels, img_resolution, img_resolution) input image
            noise_labels (torch.Tensor): Shape (batch_size,) noise (timestep) labels
            class_labels (torch.Tensor): Shape (batch_size, num_classes+1) class labels (one-hot vector)
        """
        batch_size = x.shape[0]
        assert noise_labels.shape[0] == batch_size
        assert class_labels.shape[0] == batch_size

        ## creation of noise (timestep) embedding and class embedding
        embeddings = self.get_embedding(noise_labels, class_labels)

        ## forward pass in the model
        # encoder pass
        skips = []
        for _, block in self.encoder.items():
            x = (
                block(x, embeddings)
                if isinstance(block, layers.UNetBlock)
                else block(x)
            )
            skips.append(x)

        # decoder pass
        for block_name, block in self.decoder.items():
            if "aux_norm" in block_name:
                x = block(x)
            elif "aux_conv" in block_name:
                x = block(nn.functional.silu(x))
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, embeddings)
        return x
