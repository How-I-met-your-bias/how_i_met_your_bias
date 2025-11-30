"""
Nathan Roos
"""

import os
import sys

import torch
import ml_collections

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import himyb.training.save_load as save_load


def test_save_and_load_simple():
    """
    Test on a simple example that saving and loading works
    """

    class DummyModel(torch.nn.Module):
        def __init__(self, init_weights=None):
            super(DummyModel, self).__init__()
            if init_weights is None:
                init_weights = torch.zeros(3, 3)
            self.fc = torch.nn.Linear(3, 3)
            self.fc.weight.data = init_weights

        def forward(self, x):
            return self.fc(x)

    # prepare the dummy data to save
    path_to_save_states = "./dummy_training_state.pth"
    path_to_save_configs = "./dummy_training_configs.pth"
    model = DummyModel(torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.param_groups[0]["lr"] = 77777777
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    scheduler.gamma = 498
    dataset_config = ml_collections.ConfigDict(
        {
            "dataset dummy field": "dataset dummy value",
            "dataset dummy field 2": "dataset dummy value 2",
        }
    )
    model_config = {
        "training dummy field": "training dummy value",
        "training dummy field 2": "training dummy value 2",
    }
    optim_config = {
        "optim dummy field": "optim dummy value",
        "optim dummy field 2": "optim dummy value 2",
    }
    cur_nimg = 148

    # prepare the dummy objects that will receive the loaded data
    # (they will be compared to the original objects)
    loaded_model = DummyModel()
    loaded_optimizer = torch.optim.Adam(loaded_model.parameters())
    loaded_scheduler = torch.optim.lr_scheduler.StepLR(
        loaded_optimizer, step_size=1, gamma=0.1
    )

    # save the data
    save_load.save_training_state(
        path_to_save_states,
        model,
        optimizer,
        scheduler,
    )
    save_load.save_training_configs(
        path_to_save_configs,
        dataset_config,
        model_config,
        optim_config,
        cur_nimg,
    )

    # load the data
    loaded_dataset_config, loaded_model_config, loaded_optim_config, loaded_cur_nimg = (
        save_load.load_training_configs(path_to_save_configs)
    )
    save_load.load_training_state(
        path_to_save_states,
        loaded_model,
        loaded_optimizer,
        loaded_scheduler,
    )

    # check that the loaded data is the same as the saved data
    assert loaded_model.state_dict().keys() == model.state_dict().keys()
    for (_, t1), (_, t2) in zip(
        model.state_dict().items(), loaded_model.state_dict().items()
    ):
        if isinstance(t1, torch.Tensor):
            assert torch.equal(t1, t2)
    assert loaded_optimizer.state_dict() == optimizer.state_dict()
    assert loaded_scheduler.state_dict() == scheduler.state_dict()
    assert loaded_dataset_config == dataset_config
    assert loaded_optim_config == optim_config
    assert loaded_model_config == model_config
    assert loaded_cur_nimg == cur_nimg

    # clean up
    os.remove(path_to_save_states)
    os.remove(path_to_save_configs)
