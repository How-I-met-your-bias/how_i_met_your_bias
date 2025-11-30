"""
Nathan Roos

Define functions to save and load training state
"""

import torch
import ml_collections


def save_training_state(
    path_to_save,
    model,
    optimizer,
    scheduler,
):
    """
    Save the states of the model, optimizer and scheduler to a file
    """

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path_to_save,
    )


def save_training_configs(
    path_to_save,
    dataset_config,
    model_config,
    optim_config,
    cur_nimg,
):
    """
    Save the configurations for training to a file
    """
    # Add ml_collections.ConfigDict to the list of safe globals
    # (to be able to load with weights_only=True)
    torch.serialization.add_safe_globals([ml_collections.ConfigDict])
    torch.save(
        {
            "dataset_config": dataset_config,
            "model_config": model_config,
            "optim_config": optim_config,
            "cur_nimg": cur_nimg,
        },
        path_to_save,
    )


def load_training_state(file_name, model=None, optimizer=None, scheduler=None, weights_only=True):
    """
    Load the training state from a file
    The file should have been saved with `save_training_state`
    Only load the training state if the corresponding argument is not None.
    """

    training_state = torch.load(file_name, weights_only=weights_only)
    if model is not None :
        if "model_state_dict" in training_state:
            model.load_state_dict(training_state["model_state_dict"])
        else:
            print(f"No model_state_dict key found in {file_name}")
    if optimizer is not None:
        if "optimizer_state_dict" in training_state:
            optimizer.load_state_dict(training_state["optimizer_state_dict"])
        else:
            print(f"No optimizer_state_dict key found in {file_name}")
    if scheduler is not None:
        if "scheduler_state_dict" in training_state:
            scheduler.load_state_dict(training_state["scheduler_state_dict"])
        else:
            print(f"No scheduler_state_dict key found in {file_name}")

def load_training_configs(file_name, weights_only=True):
    """
    Load the training configurations from a file
    The file should have been saved with `save_training_configs`

    Returns:
        dataset_config, model_config, optimizer_config, cur_nimg
    """
    # Add ml_collections.ConfigDict to the list of safe globals
    # (to be able to load with weights_only=True)
    torch.serialization.add_safe_globals([ml_collections.ConfigDict])

    training_state = torch.load(file_name, weights_only=weights_only)
    dataset_config = training_state["dataset_config"]
    optimizer_config = training_state["optim_config"]
    model_config = training_state["model_config"]
    cur_nimg = training_state["cur_nimg"]
    return dataset_config, model_config, optimizer_config, cur_nimg
