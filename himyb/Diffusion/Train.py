
import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler
from datasets.bffhq import BFFHQ
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets.biased_mnist import get_dataloader
from sampler.dpm_solver_pytorch import *

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Union
import os
from torchvision.utils import save_image

    
def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    print("Using device: ", device)

    if modelConfig["dataset"] == "bffhq":

        dataset = BFFHQ(
            root="./data/bffhq",
            env="train",
            bias_amount=0.995,
            transform = A.Compose([
                A.Resize(64, 64),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            return_index=False,
            class_label=None
        )
        dataloader = DataLoader(
            dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
        
    elif modelConfig["dataset"] == "bmnist":

        print("Using Biased MNIST dataset.")
        dataloader = get_dataloader(root='./data', batch_size=modelConfig["batch_size"], rho=0.9,
                            n_confusing_labels=1, train=False, num_workers=2, classes_to_use=[0, 1], 
                            pin_memory=True, resolution=(32, 32))
    
    else:
        print("Dataset not valid. Exiting.")
        return

    # num_classes = 2  # For Biased MNIST, we have two classes (0 and 1)

    # Calculate epochs and iterations
    iterations = modelConfig["iterations"]
    current_iteration = 0 
    epochs = iterations // len(dataloader) + 1
    print(f"\nTraining for {epochs} epochs = {iterations} iterations ({len(dataloader)} iterations per epoch).")

    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=epochs, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=epochs // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(epochs):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for data_dict in tqdmDataLoader:

                if modelConfig["dataset"] == "cifar10":
                    images, labels = data_dict
                elif modelConfig["dataset"] == "bmnist":
                    images, labels, _ = data_dict
                else:
                    images = data_dict['image'] 
                    labels = data_dict['class_label']

                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                loss = trainer(x_0).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        os.makedirs(modelConfig["save_weight_dir"], exist_ok=True)
        if (e + 1) % 100 == 0 or e == epochs - 1:
            torch.save(net_model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], f'ckpt_{e}_.pt'))

def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(
            T=modelConfig["T"], 
            ch=modelConfig["channel"], 
            ch_mult=modelConfig["channel_mult"], 
            attn=modelConfig["attn"],
            num_res_blocks=modelConfig["num_res_blocks"], 
            dropout=0.
        ).to(device)
        
        # Load checkpoint
        ckpt = torch.load(modelConfig["training_load_weight"], map_location=device)
        model.load_state_dict(ckpt)
        print("Model weights loaded successfully.")
        model.eval()
        
        if modelConfig.get("dpm_solver", True):
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

            # 3. Define a wrapper for the UNet model.
            # DPM-Solver uses continuous time t in [0, 1], but our UNet expects discrete timesteps.
            # This function bridges that gap.
            def model_fn(x, t):
                # Convert continuous time t to discrete timesteps
                t_discrete = (t * (modelConfig["T"] - 1)).round().long()
                # The model predicts the noise (epsilon)
                return model(x, t_discrete)

            # 4. Instantiate the DPM-Solver.
            sampler = DPM_Solver(
                model_fn=model_fn,
                noise_schedule=noise_schedule,
                algorithm_type="dpmsolver++"  # A high-performance choice
            )
            
            # -------------------- DPM-Solver Integration --------------------
        
        else:
            # If not using DPM-Solver, use the original GaussianDiffusionSampler
            sampler = GaussianDiffusionSampler(
                model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]
            ).to(device)

        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["images_to_sample"], 3, modelConfig["img_size"], modelConfig["img_size"]], 
            device=device
        )
        
        img_folder = modelConfig["sampled_dir"]
        os.makedirs(img_folder, exist_ok=True)

        # Logic to find the starting index for saving images to avoid overwriting
        existing_files = [f for f in os.listdir(img_folder) if f.startswith("sample_") and f.endswith(".png")]
        print(f"Found {len(existing_files)} existing files in {img_folder}")
        start_idx = 0
        if existing_files:
            existing_indices = [int(f.split("_")[1].split(".")[0]) for f in existing_files if f.split("_")[1].split(".")[0].isdigit()]
            if existing_indices:
                start_idx = max(existing_indices) + 1

        if modelConfig.get("dpm_solver", True):
            # 4. Sample using the DPM-Solver.
            # You can control the number of steps and the solver order.
            # Fewer steps (e.g., 15-20) are often enough for great results.
            print(f"Starting sampling with DPM-Solver...")
            dpm_steps = modelConfig.get("dpm_steps", 20)
            dpm_order = modelConfig.get("dpm_order", 1)
            
            sampledImgs = sampler.sample(
                x=noisyImage,
                steps=dpm_steps,
                order=dpm_order,
                method='multistep',
                skip_type='logSNR',
                lower_order_final=True
            )
        else:
            print(f"Starting sampling with Gaussian Diffusion Sampler...")
            sampledImgs = sampler(noisyImage)

        print("Sampling complete.")

        # Save the generated images
        for i, img in enumerate(sampledImgs):
            # Denormalize from [-1, 1] to [0, 1]
            img_to_save = torch.clamp(img * 0.5 + 0.5, 0, 1)
            save_image(img_to_save, os.path.join(img_folder, f"sample_{start_idx + i}.png"))
        
        print(f"Saved {len(sampledImgs)} new images to {img_folder}.")