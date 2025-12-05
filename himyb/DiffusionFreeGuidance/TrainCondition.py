

import os
import math
from typing import Dict
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from DiffusionFreeGuidance.DiffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer
from DiffusionFreeGuidance.ModelCondition import UNet
from DiffusionFreeGuidance.Scheduler import GradualWarmupScheduler
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import ConcatDataset
import albumentations as A
from datasets import bffhq

from datasets.biased_mnist import get_dataloader
from sampler.dpm_solver_pytorch import *

class ConcatDatasetWithAttributes(ConcatDataset):
    """Extended ConcatDataset that maintains access to original dataset attributes"""
    @property
    def num_classes(self):
        return self.datasets[0].num_classes


def prompt_dataset_split() -> str:
    """
    Prompts the user to select the dataset configuration for `waterbirds`.
    Returns:
        str: The configuration selected by the user.
    """
    print("\nSelect the dataset configuration for Waterbirds:")
    print("1. Standard dataset (5% conflict samples)")
    print("2. Ablation: Custom split with 10%, 20%, 30%, or 40% conflict samples")
    choice = input("Select an option (1 or 2): ").strip()

    if choice == "2":
        print("\nAvailable Bias-Conflict/Align splits:")
        print("10%: 203/1827")
        print("20%: 406/1624")
        print("30%: 609/1421")
        print("40%: 812/1218")
        split = input("Enter the desired percentage (10, 20, 30, 40): ").strip()
        if split in {"10", "20", "30", "40"}:
            return f"{split}%"
        else:
            print("Invalid choice! Falling back to standard dataset.")
    return "standard"


def get_dataset(modelConfig: Dict, transform) -> ConcatDataset:
    """
    Prepares the dataset based on the configuration and user input.

    Args:
        modelConfig (Dict): Configuration dictionary.
        transform: Transformations to apply to the dataset.

    Returns:
        ConcatDataset: Combined dataset for training, validation, and testing.
    """
    if modelConfig["dataset"] == "bffhq":
        dataset = bffhq.BFFHQ(root=modelConfig["data_dir"], env="train", transform=transform)

    else:
        raise ValueError(f"Dataset {modelConfig['dataset']} not supported.")

    num_classes = len(dataset.classes)

    return dataset, num_classes


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    # Data transform
    data_transform = A.Compose([
        A.Resize(modelConfig["img_size"], modelConfig["img_size"]),
        A.Normalize(normalization='standard'),
        A.pytorch.ToTensorV2()
    ])

    if modelConfig['dataset'] == 'bmnist':
        print("Using Biased MNIST dataset.")
        dataloader = get_dataloader(root='./data', batch_size=modelConfig["batch_size"], rho=0.9,
                            n_confusing_labels=1, train=False, num_workers=2, classes_to_use=[0, 1], 
                            pin_memory=True, resolution=(32, 32))
        
        num_classes = 2

    else:
        # Prepare dataset
        dataset, num_classes = get_dataset(modelConfig, data_transform)

        dataloader = DataLoader(
            dataset, batch_size=modelConfig["batch_size"], shuffle=True,
            num_workers=16, drop_last=False, pin_memory=True
        )

    # Calculate epochs and iterations
    iterations = modelConfig["iterations"]
    current_iteration = 0 
    epochs = iterations // len(dataloader) + 1
    print(f"\nTraining for {epochs} epochs = {iterations} iterations ({len(dataloader)} iterations per epoch).")

    # Initialize the model 
    net_model = UNet(T=modelConfig["T"], num_labels=num_classes, ch=modelConfig["channel"],
                    ch_mult=modelConfig["channel_mult"], num_res_blocks=modelConfig["num_res_blocks"], 
                    dropout=modelConfig["dropout"]).to(device)

    # Initialize optimizer 
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)

    cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=epochs, eta_min=0, last_epoch=-1
    )

    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                                warm_epoch=epochs // 10, after_scheduler=cosineScheduler)
                
    # Initialize the diffusion trainer
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training       
    for epoch in range(0, epochs):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for data in tqdmDataLoader:

                if modelConfig["dataset"] == "cifar10":
                    images, labels = data
                elif modelConfig["dataset"] == "bmnist":
                    images, labels, _ = data
                else:
                    images = data['image'] 
                    labels = data['class_label']
                
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device) + 1
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)
                loss = trainer(x_0, labels).sum() / b ** 2.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                current_iteration += 1
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": epoch,
                    "iteration": current_iteration,
                    "loss: ": loss.item(),
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
                
                if modelConfig["freq_save"] > 0 and current_iteration % modelConfig["freq_save"] == 0:
                    model_path = os.path.join(modelConfig["save_dir"], modelConfig["dataset"], "ckpt_" + str(current_iteration) + "_iterations.pt")
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    torch.save({
                        'model_state_dict': net_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': warmUpScheduler.state_dict(),
                        'learning_rate': warmUpScheduler.get_last_lr(),
                        'iteration': current_iteration,
                        'epoch': epoch,
                    }, model_path)

        warmUpScheduler.step()


def compute_fid(real_images: torch.Tensor, generated_images: torch.Tensor, device: str = "cuda") -> float:

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    # Update with real images
    for batch in torch.split(real_images, 16):  # Process in batches to avoid memory issues
        fid_metric.update(torch.clamp(batch, 0.0, 1.0), real=True)
    
    # Update with generated images
    for batch in torch.split(generated_images, 16):
        fid_metric.update(torch.clamp(batch, 0.0, 1.0), real=False)
    
    return fid_metric.compute().item()


class UNetIntegerTimeWrapper(torch.nn.Module):
    """
    A wrapper for the UNet model that converts the float time tensor
    from the DPM-solver (continuous) to a valid integer timestep [0, T-1],
    and handles the conditional input for classifier-free guidance.
    """
    def __init__(self, model, T: int):
        super().__init__()
        self.model = model
        self.T = T  # total number of diffusion steps

    def forward(self, x, t, condition=None):
        """
        Args:
            x: input tensor
            t: continuous timestep (float, usually in [0, 1] or [0, T])
            condition: optional class condition (long tensor or None)
        """
        # If solver gives t in [0, 1], scale to [0, T-1]
        if t.dtype.is_floating_point:
            if t.max() <= 1.0 + 1e-6:  # assume normalized
                t = t * (self.T - 1)

        # Convert to nearest integer
        t_long = torch.round(t).long()

        # Clamp to valid range [0, T-1]
        t_long = t_long.clamp(0, self.T - 1)

        return self.model(x, t_long, condition)


def eval(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # os.makedirs(os.path.join(modelConfig["sampled_dir"], modelConfig["dataset"]), exist_ok=True)
    
    data_transform = A.Compose([
        A.Resize(modelConfig["img_size"], modelConfig["img_size"]),
        A.Normalize(normalization='standard'),
        A.pytorch.ToTensorV2()
    ])

    # if modelConfig['dataset'] == 'bmnist':
    #     print("Using Biased MNIST dataset.")
    #     dataloader = get_dataloader(root='./data', batch_size=modelConfig["batch_size"], rho=0.9,
    #                         n_confusing_labels=1, train=False, num_workers=2, classes_to_use=[0, 1], 
    #                         pin_memory=True, resolution=(32, 32))

    #     num_classes = 2  # For Biased MNIST, we have two classes (0 and 1)

    # else:
    #     # Prepare dataset
    #     dataset, num_classes = get_dataset(modelConfig, data_transform)
    
    num_classes = 2

    # Load model and sampler
    model = UNet(T=modelConfig["T"], num_labels=num_classes, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                 num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    

    model_path = modelConfig["load_weights"]
    iterations = int(os.path.basename(model_path).split("_")[1])

    ckpt = torch.load(model_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    print(f"Model loaded from {model_path}")
    del ckpt

    model.eval()

    if not modelConfig["dpm-solver++"]:
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)
    
    batch_size = modelConfig["batch_size"]
    samples = modelConfig["images_to_sample"]
    if batch_size > samples: batch_size = samples
    number_of_batches = math.ceil(samples / batch_size)
    print(f"Generating {samples} samples in {number_of_batches} batches.")

    if modelConfig["dpm-solver++"]:

        # This wrapper will handle the float-to-int conversion for the time tensor.
        wrapped_model = UNetIntegerTimeWrapper(model, T=modelConfig["T"]).to(device)

        dpm_steps = modelConfig.get("dpm_steps", 20)
        dpm_order = modelConfig.get("dpm_order", 1)
    
    # Create a folder for individual images
    img_folder = modelConfig["sampled_dir"]
    os.makedirs(img_folder, exist_ok=True)

    if modelConfig["dpm-solver++"]:
        # ------------------- DPM-Solver Integration -------------------

        # 1. Create the linear beta schedule, same as in your trainer.
        # This tensor contains the variance for each discrete timestep.
        betas = torch.linspace(
            modelConfig["beta_1"], 
            modelConfig["beta_T"], 
            modelConfig["T"], 
            device=device
        )

        # 2. Create the noise schedule by passing the discrete betas directly.
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

        # -------------------- DPM-Solver Integration --------------------

    # Compute FID for each class
    for class_label in range(num_classes):

        class_labels = torch.full((batch_size,), class_label+1, dtype=torch.long, device=device)
        uncond_class_labels = torch.zeros((batch_size,), dtype=torch.long, device=device)

        if modelConfig["dpm-solver++"]:
            ## 2. Convert your discrete-time `model` to the continuous-time
            ## noise prediction model. Here is an example for a diffusion model
            ## `model` with the noise prediction type ("noise") .
            model_fn = model_wrapper(
                wrapped_model,
                noise_schedule,
                model_type="noise",  # or "x_start" or "v" or "score"
                # model_kwargs=model_kwargs,
                guidance_type="classifier-free",
                condition=class_labels,
                unconditional_condition=uncond_class_labels,
                guidance_scale=modelConfig["w"],
            )

            # 4. Instantiate the DPM-Solver.
            sampler = DPM_Solver(
                model_fn=model_fn,
                noise_schedule=noise_schedule,
                algorithm_type="dpmsolver++"  # A high-performance choice
            )
        
        print(f"Processing class {class_label}...")
        # real_images = []
        # generated_images = []
        
        # Collect real images for the current class
        # dataloader = DataLoader(dataset, batch_size=batch_size)
        # for real_batch in dataloader:

        #     if modelConfig["dataset"] == "cifar10":
        #         mask = real_batch[1] == class_label
        #         real_images.extend(real_batch[0][mask])
        #     elif modelConfig["dataset"] == "bmnist":
        #         mask = real_batch[1] == class_label
        #         real_images.extend(real_batch[0][mask])
        #     else:
        #         mask = real_batch['class_label'] == class_label
        #         real_images.extend(real_batch['image'][mask])

        #     if len(real_images) >= samples:
        #         break
        # real_images = torch.stack(real_images)[:samples].to(device)

        # Generate images for the current class
        for i in range(number_of_batches):
            with torch.no_grad():
                noisy_images = torch.randn(
                    size=[batch_size, 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
                if modelConfig["dpm-solver++"]:
                    sampledImgs = sampler.sample(
                        x=noisy_images,
                        steps=dpm_steps,
                        order=dpm_order,
                        method='multistep',
                        skip_type='logSNR',
                        lower_order_final=True
                    )
                else:
                    sampledImgs = sampler(noisy_images, class_labels, start_t=modelConfig["T"])
           
            existing_files = [f for f in os.listdir(img_folder) if f.startswith("sample_") and f.endswith(".png")]
            print(f"Found {len(existing_files)} existing files in {img_folder}")
            if existing_files or modelConfig["continue_from_existing"]:
                existing_indices = [int(f.split("_")[1].split(".")[0]) for f in existing_files if f.split("_")[1].split(".")[0].isdigit()]
                start_idx = max(existing_indices) + 1 if existing_indices else 0
            else:
                start_idx = 0

            for i, img in enumerate(sampledImgs):
                img_to_save = torch.clamp(img * 0.5 + 0.5, 0, 1)
                save_image(img_to_save, os.path.join(img_folder, f"sample_{start_idx + i}_class{class_label}.png"))
