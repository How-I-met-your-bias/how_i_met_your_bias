"""
Nathan Roos

Example script to generate samples from a diffusion model trained on Biased MNIST.
It loads the model and generates samples for each class, saving the results in a matrix.
The matrix is saved as a CSV file, and a configuration file is created with the parameters used for sampling.

The root_dir should contain subfolders for each run, and have the following structure:
root_dir/
├── run_name_1/
│   ├── checkpoints/
│   │   ├── *end_training*_configs.pth
│   │   ├── *end_training*_states.pth
│   ├── generated_color_matrix.csv          //output file containing the counts for each class-bias combination
│   ├── generated_color_matrix_config.txt   //output file containing the parameters used for sampling
├── run_name_2/
| ...

In this script, we generate samples from each of the classes as well as from the unconditional class
(class token 0 in the model). For each of the sample, we compute the color index of the generated image,
which corresponds to the bias label.
For instance, we generate a 1, and see that it is colored red, which is the color of the class 0.
We add 1 at the position (1, 0) in the matrix of counts.

As an illustration, the matrix should look like this if we generate 100 samples from each class
and that rho_model=0.9 (ie there is a 90% proportion of bias-aligned samples in the generated samples):
(the headers are here for clarity, they are not in the output file)

                bias 0   bias 1   bias 2   (legacy column)
class 0         90,       5,          5,         0
class 1         5,        90,         5,         0
class 2         5,        5,          90,        0
unconditional   33,       33,         34,        0

The last column is a legacy column (should be carefully removed) that is always filled with zeros.

A real example of output for 2 classes is :
3.439000000000000000e+03,5.610000000000000000e+02,0.000000000000000000e+00
9.200000000000000000e+01,3.908000000000000000e+03,0.000000000000000000e+00
1.447000000000000000e+03,2.553000000000000000e+03,0.000000000000000000e+00

If you don't plan on using the the unconditional class (for instance if you only want to
compute rho_model), you should adapt the code to not generate samples from the unconditional class.
This will save compute.

FURTHER WORK: in this script, we do not control the quality of the generated samples. If you seek
to control this metric, you should compute it in this script, in the sampling loop, at the same time
as the bias label (color index) is computed.
"""

import os

import numpy as np
import torch
import tqdm

import himyb.training.save_load as save_load
import himyb.models.ddpmpp as ddpmpp
import himyb.models.preconditioning as precond
import himyb.sampler.sampler as sampler
import himyb.datasets.biased_mnist_utils as biased_mnist_utils


def dump_dict_to_file(dump_dict, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for key, value in dump_dict.items():
            f.write(f"{key}: {value}\n")


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


ROOT_DIR = "/home/ids/nroos-23/experiments/himyb.llh/bmnist/2_classes_small_dataset"
RUN_NAMES = [
    "bmnist_class_size=500_rho=0.9",
    "bmnist_class_size=1000_rho=0.9",
    "bmnist_class_size=1500_rho=0.9",
    "bmnist_class_size=2000_rho=0.9",
    "bmnist_class_size=2500_rho=0.9",
    "bmnist_class_size=3000_rho=0.9",
    "bmnist_class_size=3500_rho=0.9",
    "bmnist_class_size=4000_rho=0.9",
    "bmnist_class_size=4500_rho=0.9",
    "bmnist_class_size=5000_rho=0.9",
    "bmnist_class_size=5500_rho=0.9",
    "bmnist_class_size=6000_rho=0.9",
    "bmnist_class_size=6332_rho=0.9",
]

# number of samples to generate per class per model
NUM_CLASSES = 2
BATCH_SIZE_PER_CLASS = 800
BATCH_SIZE = BATCH_SIZE_PER_CLASS * (NUM_CLASSES + 1)
BATCH_NUMBER = 5
NUM_STEPS = 200
print(f"Number of images per model {BATCH_SIZE*BATCH_NUMBER}")
assert torch.cuda.is_available(), "CUDA is not available. Please run on a GPU."
DEVICE = "cuda"

# parameters of stochastic sampler
RHO = 7
S_CHURN = 40
S_MIN = 0.05
S_MAX = 50
S_NOISE = 1.003
SIGMA_MIN = 0.002
SIGMA_MAX = 80

for run_name in RUN_NAMES:

    ## load or create matrix of class-bias counts
    run_dir = os.path.join(ROOT_DIR, run_name)
    save_file_name = "generated_color_matrix"
    matrix_file = os.path.join(run_dir, f"{save_file_name}.csv")
    # if the matrix file already exists, we load it and we will add the new counts to it
    # otherwise, we create a new matrix filled with zeros
    if os.path.exists(matrix_file):
        print(f"Matrix file exists for run {run_name}. Starting from there.")
        matrix = np.loadtxt(matrix_file, delimiter=",").astype(int)
    else:
        matrix = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1)).astype(int)

    ## create model and load weights
    ckpt_path = os.path.join(run_dir, "checkpoints")
    model_config_file_path = get_config_file_path(ckpt_path)
    model_state_file_path = get_state_file_path(ckpt_path)
    _, model_config, _, _ = save_load.load_training_configs(model_config_file_path)
    wrapped_model = ddpmpp.DDPMPP(**model_config)
    diffusion_model = precond.EDMPrecond(wrapped_model)
    save_load.load_training_state(model_state_file_path, diffusion_model)
    diffusion_model.eval().to(DEVICE)

    # create class labels: 0 is the unconditional class, class 1 is class 0 in the dataset,
    # class 2 is class 1 in the dataset, etc.
    class_labels = torch.arange(NUM_CLASSES + 1).repeat(BATCH_SIZE_PER_CLASS).to(DEVICE)
    shape = (
        BATCH_SIZE,
        diffusion_model.in_channels,
        diffusion_model.img_resolution,
        diffusion_model.img_resolution,
    )

    # sample images and compute color index
    for batch_idx in tqdm.tqdm(range(BATCH_NUMBER), desc=f"Run {run_name}"):
        # generate samples
        sampling_result = sampler.stoch_edm_sampler(
            diffusion_model,
            class_labels=class_labels,
            shape=shape,
            num_steps=NUM_STEPS,
            device=DEVICE,
            rho=RHO,
            s_churn=S_CHURN,
            s_min=S_MIN,
            s_max=S_MAX,
            s_noise=S_NOISE,
            sigma_min=SIGMA_MIN,
            sigma_max=SIGMA_MAX,
        )
        sampled_imgs = sampling_result["generated_imgs"]

        # get color index from batch
        color_idx = (
            biased_mnist_utils.get_color_idx_from_batch(sampled_imgs).cpu().numpy()
        )

        # add class-bias counts to matrix
        np.add.at(matrix, (class_labels.cpu().numpy() - 1, color_idx), 1)

    # save matrix
    np.savetxt(matrix_file, matrix, delimiter=",")
    dump_dict_to_file(
        {
            "run_name": run_name,
            "path": str(run_dir),
            "num_steps": NUM_STEPS,
            "batch_size": BATCH_SIZE,
            "batch_number": BATCH_NUMBER,
            "num_classes": NUM_CLASSES,
            "rho": RHO,
            "s_churn": S_CHURN,
            "s_min": S_MIN,
            "s_max": S_MAX,
            "s_noise": S_NOISE,
            "sigma_min": SIGMA_MIN,
            "sigma_max": SIGMA_MAX,
            "num_samples": BATCH_SIZE * BATCH_NUMBER,
        },
        os.path.join(run_dir, f"{save_file_name}_config.txt"),
    )
