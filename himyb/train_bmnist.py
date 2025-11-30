"""
Nathan Roos

Training script
To run it : python -m himyb.train
"""

import os
import random

import click

import himyb.training.training_loop as training_loop
import himyb.wandb_utils.wandb_utils as wandb_utils
import himyb.configs.biased_mnist.default_dataset_config as bm_def_dataset_conf
import himyb.configs.biased_mnist.default_ddpmpp_config as bm_def_ddpmpp_conf
import himyb.configs.biased_mnist.default_optimizer_config as bm_def_optimizer_conf
import himyb.configs.waterbirds.default_dataset_config as wb_def_dataset_conf
import himyb.configs.waterbirds.default_ddpmpp_config as wb_def_ddpmpp_conf
import himyb.configs.waterbirds.default_optimizer_config as wb_def_optimizer_conf


@click.command()
@click.option(
    "--rho",
    help="Rho parameter for the dataset (correlation between target and bias)",
    type=float,
    required=True,
)
@click.option(
    "--class_size",
    help="Number of samples per class (if not specified, use the maximum)",
    type=int,
    required=False,
    default=None,
)
def main(**kwargs):
    """
    Called when the script is run
    """
    dataset_config = bm_def_dataset_conf.get_default_dataset_config()
    dataset_config.rho = kwargs["rho"]
    dataset_config.classes_to_use = [0, 1]
    dataset_config.n_confusing_labels = 1
    dataset_config.class_size = kwargs["class_size"]
    dataset_config.root = "/home/ids/nroos-23/data/biased_mnist/"

    model_config = bm_def_ddpmpp_conf.get_default_ddpmpp_config()
    model_config.label_dropout = 0.2
    model_config.label_dim = 3
    model_config.num_blocks = 2
    model_config.model_channels = 64

    optimizer_config = bm_def_optimizer_conf.get_default_optimizer_config()
    optimizer_config.lr = 5e-5

    run_dir = "/home/ids/nroos-23/experiments/cond_llh/bmnist/2_classes_small_dataset"
    run_name = f"bmnist_class_size={dataset_config.class_size}_rho={dataset_config.rho}"
    wandb_wrapper = wandb_utils.WandbWrapper(
        use_wandb=True,
        run_name=run_name,
        job_type=f"train {dataset_config.name}",
    )

    training_loop.training_loop(
        run_dir=os.path.join(run_dir, run_name),
        wandb_wrapper=wandb_wrapper,
        batch_size=512,
        model_config=model_config,
        optimizer_config=optimizer_config,
        dataset_conf=dataset_config,
        save_state_every_min=10,
        seed=random.randint(0, 2**32 - 1),
        total_kimg=6000,
        save_ckpt_every_kimg=3000,
        report_loss_every_kimg=5,
        check_ema_loss_every_kimg=10,
        use_ema=True,
        ema_halflife_kimg=150,
        ema_rampup_ratio=0.05,
    )


if __name__ == "__main__":
    main()
