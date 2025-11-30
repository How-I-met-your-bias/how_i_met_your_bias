"""
Nathan Roos

Provide utility functions to determine the color of the background of colored MNIST digits.
Useful for determining the bias label of a generated image.
"""

import torch


# list of colors associated with the digits in BiasedMNIST
# it is cleaner to import this from the BiasedMNIST class
COLOUR_MAP = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [225, 225, 0],
    [225, 0, 225],
    [0, 255, 255],
    [255, 128, 0],
    [255, 0, 128],
    [128, 0, 255],
    [128, 128, 128],
]

# list of color names associated with the digits in BiasedMNIST
# it is cleaner to import this from the BiasedMNIST class
COLOUR_NAMES = [
    "red",
    "green",
    "blue",
    "yellow",
    "magenta",
    "cyan",
    "orange",
    "rose",
    "electric violet",
    "grey",
]


def get_color_from_batch(batch: torch.Tensor, threshold: float = 0.7):
    """
    Compute the average RGB color of all images in a batch.

    We define the color of an image of a colored digit as the average of the rgb values of the
    non-white pixels (since the background is colored and the digits are white).

    A pixel is considered non-white if any of the RGB channels (with values in [-1,1])
    is below the threshold.

    Args:
        batch (torch.Tensor): (N, 3, H, W) tensor with values in [-1,1] (where N is the batch size, \
            H is the height and W is the width)
        threshold (torch.Tensor): threshold to determine non-white pixels
    Returns:
        avg_rgb (torch.Tensor): (N,3) tensor with average RGB of each image in the batch
    """
    batch = (batch + 1) / 2
    mask_non_white = (
        (batch[:, 0] < threshold)
        | (batch[:, 1] < threshold)
        | (batch[:, 2] < threshold)
    )
    avg_rgb = [
        batch[i, :, mask_non_white[i]].mean(dim=1) for i in range(batch.shape[0])
    ]
    avg_rgb = torch.stack(avg_rgb, dim=0)
    return avg_rgb


def get_distance_to_colors(
    batch: torch.Tensor, colors: torch.Tensor, threshold: float = 0.7
) -> torch.Tensor:
    """
    Compute the distance in RGB space between the color of each image in the batch
    and a set of colors.

    Args:
        batch (torch.Tensor): (N,Ch, H, W) tensor of images with values in [-1,1]
        colors (torch.Tensor): (Co,3) tensor with average RGB values for non-white pixels (in [0,255])
        threshold (float): threshold to determine non-white pixels (a pixel is considered non-white \
            if any of the RGB channels is below the threshold)
    Returns:
        distances (torch.Tensor): (N,Co) tensor with distance to each color
    """
    batch_colors = get_color_from_batch(batch, threshold=threshold) * 255
    colors.to(batch_colors.device)
    distances = (
        ((batch_colors.unsqueeze(1) - colors.unsqueeze(0)) ** 2).sum(dim=2).sqrt()
    )
    return distances


def get_color_idx_from_batch(batch: torch.Tensor, allowed_colors=None) -> torch.Tensor:
    """
    For each image in the batch, find the index of the closest color
    from the COLOUR_MAP (it corresponds to the bias label of the image).

    Args:
        img_batch (torch.Tensor): a batch of images of shape (N, C,H,W)
        allowed_colors (list[int] | None): a list of color indices to consider, \
            if None all colors are considered
    Returns:
        color_idx (torch.Tensor): a tensor of shape (N,) with the index of the closest color \
        from the COLOUR_MAP for each image in the batch
    """
    ground_truth_colors = torch.tensor(COLOUR_MAP, device=batch.device)
    distances = get_distance_to_colors(batch, ground_truth_colors)

    # if there is only a restricted set of colors to consider,
    # we create a mask to ignore the other colors
    # and set their distances to inf so that they are not selected as the closest color
    if allowed_colors is not None:
        # Create a mask where allowed_colors are False
        mask = torch.ones_like(distances, dtype=torch.bool)
        mask[:, allowed_colors] = False

        # Set distances to inf where mask is True
        distances = distances.masked_fill(mask, float("inf"))

    color_idx = distances.argmin(dim=1)
    return color_idx


def get_color_name_from_idx(color_idx: torch.Tensor) -> list[str]:
    """
    For each color index in the tensor color_idx, return the corresponding color name.

    Args:
        color_idx (torch.Tensor): a batch of color indices of shape (N,)
    Returns:
        _ (list[str]): a list of color names corresponding to the indices in color_idx
    """
    color_names = COLOUR_NAMES
    return [color_names[i] for i in color_idx]


def get_color_name_from_batch(batch: torch.Tensor) -> list[str]:
    """
    For each image in the batch, find the name of the closest color
    from the COLOUR_MAP (it corresponds to the bias label of the image).

    Args:
        img_batch (torch.Tensor): a batch of images of shape (N, C,H,W)
    Returns:
        _ (list[str]): a list of color names corresponding to the closest color for each image
    """
    color_idx = get_color_idx_from_batch(batch)
    color_names = get_color_name_from_idx(color_idx)
    return color_names
