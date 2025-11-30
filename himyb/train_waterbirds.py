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
def main(**kwargs):
    """
    Called when the script is run
    """
    dataset_config = wb_def_dataset_conf.get_default_dataset_config()
    dataset_config.rho = kwargs["rho"]
    dataset_config.root = "/home/ids/nroos-23/data/waterbirds/"

    model_config = wb_def_ddpmpp_conf.get_default_ddpmpp_config()

    optimizer_config = wb_def_optimizer_conf.get_default_optimizer_config()
    optimizer_config.lr = 2e-4

    run_dir = "/home/ids/nroos-23/experiments/cond_llh/waterbirds/"
    run_name = f"rho={dataset_config.rho}_lr={optimizer_config.lr}"
    wandb_wrapper = wandb_utils.WandbWrapper(
        use_wandb=True,
        run_name=run_name,
        job_type=f"train {dataset_config.name}",
    )

    training_loop.training_loop(
        run_dir=os.path.join(run_dir, run_name),
        wandb_wrapper=wandb_wrapper,
        batch_size=64,
        model_config=model_config,
        optimizer_config=optimizer_config,
        dataset_conf=dataset_config,
        save_state_every_min=30,
        seed=random.randint(0, 2**32 - 1),
        total_kimg=15000,
        save_ckpt_every_kimg=5000,
        report_loss_every_kimg=10,
        check_ema_loss_every_kimg=20,
        use_ema=True,
        ema_halflife_kimg=200,
    )


if __name__ == "__main__":
    main()
