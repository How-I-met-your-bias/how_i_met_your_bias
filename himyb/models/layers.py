"""
Modified by Nathan Roos

Utility layers
Taken from the implementation of Karras 2022

Previous work (https://github.com/NVlabs/edm) copyright mention :

    Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    This work is licensed under a Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License.
    You should have received a copy of the license along with this
    work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import torch
from torch.nn.functional import silu
import numpy as np


def weight_init(shape, mode, fan_in, fan_out):
    """
    Initialize weights for a given shape and mode

    Args:
        shape (tuple): Shape of the weights
        mode (str): Initialization mode (xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal)
        fan_in (int): Number of input units
        fan_out (int): Number of output units
    """
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class PositionalEmbedding(torch.nn.Module):
    """
    Custom neural network layer that creates positional embeddings
    given an input tensor of positions
    """

    def __init__(self, num_channels, max_positions=10000, endpoint=True):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        """
        Args :
            x (torch.Tensor): Shape (batch_size,)

        Returns :
            output (torch.Tensor): Shape (batch_size, num_channels)
        """
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = torch.outer(x, freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class Linear(torch.nn.Module):
    """
    Conveniance linear layer that initializes the weights and biases
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(
            weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Shape (batch_size, in_features)
        Returns:
            output (torch.Tensor): Shape (batch_size, out_features)
        """
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


class Conv2d(torch.nn.Module):
    """
    Convenience 2D convolutional layer that initializes the weights and biases
    and can perform upsampling or downsampling with box filter
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        bias=True,
        up=False,
        down=False,
        resample_filter=None, # default is [1, 1]
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        assert not (up and down)
        if resample_filter is None:
            resample_filter = [1, 1]
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel,
            fan_out=out_channels * kernel * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs)
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = torch.outer(f, f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = (
            self.resample_filter.to(x.dtype)
            if self.resample_filter is not None
            else None
        )
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.up:
            x = torch.nn.functional.conv_transpose2d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=f_pad,
            )
        if self.down:
            x = torch.nn.functional.conv2d(
                x,
                f.tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=f_pad,
            )
        if w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


class GroupNorm(torch.nn.Module):
    """
    Convenience group normalization layer that initializes the weights and biases to 1 and 0 respectively
    and compute the correct number of groups
    """

    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Shape (batch_size, num_channels, height, width)
        Returns:
            output (torch.Tensor): Shape (batch_size, num_channels, height, width)
        """
        x = torch.nn.functional.group_norm(
            x,
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x


class AttentionOp(torch.autograd.Function):
    """
    Custom autograd function for the self-attention operation
    Computes attention weights , i.e., softmax(Q^T * K).
    Performs all computation using FP32, but uses the original datatype for
    inputs/outputs/gradients to conserve memory.

    (kept as in the original implementation)
    """

    @staticmethod
    def forward(ctx, q, k):
        w = (
            torch.einsum(
                "ncq,nck->nqk",
                q.to(torch.float32),
                (k / np.sqrt(k.shape[1])).to(torch.float32),
            )
            .softmax(dim=2)
            .to(q.dtype)
        )
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(
            grad_output=dw.to(torch.float32),
            output=w.to(torch.float32),
            dim=2,
            input_dtype=torch.float32,
        )
        dq = torch.einsum("nck,nqk->ncq", k.to(torch.float32), db).to(
            q.dtype
        ) / np.sqrt(k.shape[1])
        dk = torch.einsum("ncq,nqk->nck", q.to(torch.float32), db).to(
            k.dtype
        ) / np.sqrt(k.shape[1])
        return dq, dk


class UNetBlock(torch.nn.Module):
    """
    Convenience residual block layer that initializes the weights and biases

    Perform all the operations of a residual block by going through the layers:
        1. GroupNorm (norm0) (+ SiLU)
        2. Conv2d (conv0)
        3. add embedding
        4. GroupNorm (norm1) (+ SiLU)
        5. Conv2d (conv1)
        6. add x (linear transformed by self.skip if necessary) # skip connection
        7. Multiply by skip_scale

        If self_attention :
            8. add self attention and scale by skip_scale
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        resample_filter=None,  # default is [1, 1]
        resample_proj=False,
        init=None,  # default is {}
        init_zero=None,  # default is {init_weight=0}
        init_attn=None,
    ):
        # default values
        if resample_filter is None:
            resample_filter = [1, 1]
        if init is None:
            init = {}
        if init_zero is None:
            init_zero = dict(init_weight=0)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = num_heads if attention and num_heads else 0
        self.dropout = dropout
        self.skip_scale = skip_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            **init,
        )
        self.affine = Linear(
            in_features=emb_channels,
            out_features=out_channels,
            **init,
        )  # affine transformation of the embedding to the same size as the output
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                **init,
            )

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * 3,
                kernel=1,
                **(init_attn if init_attn is not None else init),
            )
            self.proj = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel=1,
                **init_zero,
            )

    def forward(self, x, embedding):
        """
        Args:
            x (torch.Tensor): Shape (batch_size, in_channels, height, width)
            embedding (torch.Tensor): Shape (batch_size, emb_channels)
        """

        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(embedding).unsqueeze(2).unsqueeze(3).to(x.dtype)
        x = silu(self.norm1(x.add_(params)))

        x = self.conv1(
            torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        )
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = (
                self.qkv(self.norm2(x))
                .reshape(
                    x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1
                )
                .unbind(2)
            )
            w = AttentionOp.apply(q, k)
            a = torch.einsum("nqk,nck->ncq", w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x
