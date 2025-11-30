"""
Nathan Roos

"""

import torch
import matplotlib.pyplot as plt

import himyb.sampler.sampler as sampler
import himyb.misc_utils as misc_utils


def sample_and_save(
    path_to_save: str,
    num_imgs_per_class: int,
    model: torch.nn.Module,
    num_steps: int,
    device,
):
    """
    Generate label_dim*num_imgs_per_class images from the model and save them to path_to_save in one single file.

    Args:
        path_to_save (str): Path to save the generated images
        num_imgs_per_class (int): Number of images to generate from each class
        model (torch.nn.Module): Model to sample from
        num_steps (int): Number of integration steps of the ODE
    """

    def rescale(x):
        """maps [-1, 1] to [0, 1]"""
        return x / 2 + 0.5

    label_dim = model.label_dim if model.label_dim != 0 else 1
    class_labels = torch.ones(
        (label_dim, num_imgs_per_class), dtype=torch.long
    ) * torch.arange(label_dim).reshape(-1, 1)
    class_labels = class_labels.reshape(-1)
    tot_nimgs = label_dim * num_imgs_per_class
    result = sampler.det_edm_sampler(
        model=model,
        num_steps=num_steps,
        class_labels=class_labels,
        shape=(
            tot_nimgs,
            model.in_channels,
            model.img_resolution,
            model.img_resolution,
        ),
        device=device,
    )
    generated_imgs = result["generated_imgs"]

    # Save the images
    fig, axes = plt.subplots(
        label_dim,
        (num_imgs_per_class + 1),
        figsize=(num_imgs_per_class * 1, label_dim * 1),
    )
    for i in range(label_dim):
        for j in range(num_imgs_per_class + 1):
            ax = axes[i, j]
            if i == 0 and j == 0:
                ax.set_title(f"NFE={result['NFE']}")
            if j == 0:
                ax.text(0.5, 0.5, f"cond:{i}", fontsize=8, ha="center")
                ax.axis("off")
            else:
                img_idx = i * num_imgs_per_class + j - 1
                with misc_utils.DisableImshowWarning():
                    ax.imshow(
                        rescale(generated_imgs[img_idx]).permute(1, 2, 0).cpu().numpy()
                    )
                ax.axis("off")
    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.close(fig)
