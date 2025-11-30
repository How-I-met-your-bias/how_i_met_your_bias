"""
Sample images from BFFHQ trained models and save them to a folder.

Assumes the folder to have the following structure:

root_folder/
├── checkpoints/
│   ├── *end_training*_configs.pth
│   ├── *end_training*_states.pth

The images will be stored in /root_folder/generated_images/<custom name indicating the parameters>/

Example usage:
python -m himyb.sampling_bffhq --root_folder ~/experiments/cond_llh/bffhq/rho=0.95_lr=1e-06_from_pretrained --n_steps 10 --s_churn 80.0 --batch_size 256 --n_samples 5000 --guidance_scale 0.0
"""

import os

import argparse
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt


import himyb.training.save_load as save_load
import himyb.models.ddpmpp as ddpmpp
import himyb.models.preconditioning as precond
import himyb.sampler.sampler as sampler


def get_config_file_path(folder_path):
    """
    Get the path to the config file that contains 'end_training' in the given folder.
    """
    for file_name in os.listdir(folder_path):
        if "end_training" in file_name and file_name.endswith("_configs.pth"):
            return os.path.join(folder_path, file_name)
    return None


def get_state_file_path(folder_path):
    """
    Get the path to the state file that contains 'end_training' in the given folder.
    """
    for file_name in os.listdir(folder_path):
        if "end_training" in file_name and file_name.endswith("_states.pth"):
            return os.path.join(folder_path, file_name)
    return None


def save_imgs(imgs, class_labels, folder):
    """
    Save images to a specified folder, naming them according to thei class label.
    The images are named 'img_{index}_{class_label}.png', where index is a number
    chosen such that no other image in the folder has the same name.


    Args:
        imgs (torch.Tensor): A batch of images with shape (B, C, H, W).
        class_labels (torch.Tensor): A tensor containing the class labels for each image.
        folder (str): The folder where the images will be saved.
    """
    os.makedirs(folder, exist_ok=True)

    cur_idx = len(os.listdir(folder))
    for i, img in enumerate(imgs):
        img = ((img.permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)
        plt.imsave(
            os.path.join(
                folder, f"img_{i+cur_idx:05d}_{int(class_labels[i].item())}.png"
            ),
            img,
        )


parser = argparse.ArgumentParser(
    description="Sample images from BFFHQ trained models and save them to a folder."
)
parser.add_argument(
    "--root_folder",
    type=str,
    required=True,
    help="Path to the folder containing the model checkpoints and the generated images",
)
parser.add_argument("--n_steps", type=int, required=True, help="Number of steps to use")
parser.add_argument(
    "--s_churn",
    type=float,
    required=True,
    help="amount of noise to add to the prediction at each step",
)
parser.add_argument(
    "--n_samples", type=int, required=True, help="Number of samples to generate"
)
parser.add_argument(
    "--batch_size", type=int, default=64, help="Batch size for generating samples"
)
parser.add_argument(
    "--guidance_scale",
    type=float,
    default=0.0,
    help="Scale for classifier-free guidance (0.0 means no guidance)",
)
args = parser.parse_args()

ROOT_DIR = args.root_folder
N_SAMPLES = args.n_samples
BATCH_SIZE = args.batch_size
N_BATCHES = N_SAMPLES // BATCH_SIZE + 1
EFFECTIVE_N_SAMPLES = N_BATCHES * BATCH_SIZE

# hyperparameters for the sampling (we could modify the code to make them all command line arguments)
N_STEPS = args.n_steps
GUIDANCE_SCALE = args.guidance_scale
S_CHURN = args.s_churn
S_MIN = 0.01
S_MAX = 80
S_NOISE = 1.003

# Print all parameters for debugging and reference
print("=== Parameters ===")
print(f"Root directory: {ROOT_DIR}")
print(f"Number of samples (effective): {EFFECTIVE_N_SAMPLES}")
print(f"Number of batches: {N_BATCHES}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Number of steps: {N_STEPS}")
print(f"S churn: {S_CHURN}")
print(f"S min: {S_MIN}")
print(f"S max: {S_MAX}")
print(f"S noise: {S_NOISE}")
print(f"Guidance scale: {GUIDANCE_SCALE}")
print("=================")


device = torch.device("cuda")

## Load the model and its configuration
config_path = get_config_file_path(os.path.join(ROOT_DIR, "checkpoints"))
if config_path is None:
    raise ValueError("No config file found in the checkpoints folder.")

states_path = get_state_file_path(os.path.join(ROOT_DIR, "checkpoints"))
if states_path is None:
    raise ValueError("No state file found in the checkpoints folder.")

dataset_conf, model_conf, optim_config, cur_img = save_load.load_training_configs(
    config_path
)

internal_model = ddpmpp.DDPMPP(**model_conf)
model = (
    precond.EDMPrecond(
        model=internal_model,
    )
    .eval()
    .to(device)
)
save_load.load_training_state(states_path, model=model)


## PREPARE FOLDER FOR SAVING IMAGES
os.makedirs(os.path.join(ROOT_DIR, "generated_images"), exist_ok=True)
folder_name = f"n_steps={N_STEPS}-s_churn={S_CHURN}-batch_size={BATCH_SIZE}-guidance_scale={GUIDANCE_SCALE}-s_min={S_MIN}-s_max={S_MAX}-s_noise={S_NOISE}"
save_img_folder = os.path.join(ROOT_DIR, "generated_images", folder_name)
os.makedirs(save_img_folder, exist_ok=True)

## SAMPLING
n_imgs_per_class = BATCH_SIZE // (model.label_dim - 1)

# the class labels in the dataset range from 0 to nb_classes - 1, but we need to start from 1 
# since the token 0 is reserved for the unconditional class in the implementation of the model, 
# so we add 1 to the class labels
class_labels = (
    torch.arange(BATCH_SIZE, device=device, dtype=torch.long) // n_imgs_per_class
) + 1
for i_batch in tqdm.tqdm(range(N_BATCHES), desc=folder_name):
    shape = (BATCH_SIZE, model.in_channels, model.img_resolution, model.img_resolution)
    result = sampler.stoch_edm_sampler(
        model=model,
        shape=shape,
        class_labels=class_labels,
        num_steps=N_STEPS,
        s_churn=S_CHURN,
        device=device,
        s_max=S_MAX,
        s_min=S_MIN,
        s_noise=S_NOISE,
        guidance_scale=GUIDANCE_SCALE,
        use_heun=True,
        use_unconditional_model=False,
        return_history=False,
    )
    imgs = result["generated_imgs"]

    # Save the images to the folder
    save_imgs(imgs, class_labels - 1, save_img_folder)
