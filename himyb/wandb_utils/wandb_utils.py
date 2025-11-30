"""
Nathan Roos

This file contains a wrapper for the wandb library
to make it easier to use in the project (you can deactivate the use of wandb such that the call to this
class will do nothing).
It loads the wandb API key and project name from a yaml file
and provides a simple interface to initialize a run, log data, and stop the run.
The yaml file should be at the root of the project and should be named wandb_info.yml.
The file should contain the following keys:
- PROJECT_NAME: The name of the wandb project
- WANDB_API_KEY: The wandb API key
"""

import time

import yaml
import wandb


FILE_WITH_WANDB_INFO = "./wandb_info.yml"


class WandbWrapper:
    """
    Wraps the wandb library and provides a simple interface to initialize a run
    and log data.
    """

    def __init__(
        self, use_wandb: bool = True, run_name: str = None, job_type: str = None
    ):
        """
        Args:
            use_wandb (bool): If True use wandb, else this class will do nothing
            run_name (str): Name of the run
            job_type (str): Type of the job
        """
        self.use_wandb = use_wandb
        if not self.use_wandb:
            print("Not using wandb")
            return
        self.run_name = run_name
        self.job_type = job_type

        # Required keys in the wandb_info.yml file
        self.required_keys = ["PROJECT_NAME", "WANDB_API_KEY"]

        self.is_running = False
        self._run = None

        stored_config = self.load_config()
        self.project_name = stored_config["PROJECT_NAME"]
        wandb.login(key=stored_config["WANDB_API_KEY"])
        del stored_config

    def load_config(self):
        """
        Load the config from FILE_WITH_WANDB_INFO
        and check if all required keys are present
        """
        stored_config = None
        with open(FILE_WITH_WANDB_INFO, encoding="utf-8") as stream:
            try:
                stored_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(f"Error loading {FILE_WITH_WANDB_INFO}")
                print(exc)
                self.use_wandb = False
                return

        if not all(key in stored_config.keys() for key in self.required_keys):
            print(f"Missing keys in {FILE_WITH_WANDB_INFO}")
            print(f"Required keys: {self.required_keys}")
            self.use_wandb = False
        return stored_config

    def init_run(self, name=None, config=None):
        """
        Initialize a wandb run"
        """
        if not self.use_wandb:
            return

        kwargs = {"project": self.project_name}
        if self.run_name is not None:
            kwargs["name"] = self.run_name
        if self.job_type is not None:
            kwargs["job_type"] = self.job_type
        if name is not None:
            kwargs["name"] = name
        if config is not None:
            kwargs["config"] = config
        self._run = wandb.init(**kwargs)
        self.is_running = True

    def stop_run(self):
        """
        Stop the current run
        """
        if not self.use_wandb:
            return

        if self.is_running:
            self._run.finish()
            self.is_running = False

    def log(self, data):
        """
        Log data to the current run
        """
        if not self.use_wandb:
            return

        if not self.is_running:
            print("No run is running")
            return

        self._run.log(data)


if __name__ == "__main__":
    wandb_wrapper = WandbWrapper(
        use_wandb=True, run_name="wandb_test_run", job_type="wandb_test_job"
    )
    wandb_wrapper.init_run(config={"test": 1})
    print("Running for 5 seconds...")
    time.sleep(5)
    wandb_wrapper.stop_run()
