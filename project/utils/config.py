import os
import yaml
import time
import random
import string
from .utils import dict_merge, make_valid_dir

dirname = os.path.dirname
config_path = os.path.join(dirname(dirname(__file__)), "config", "config.yaml")
with open(config_path, "r") as f:
    config_base = yaml.load(f)


def post_process_config(config):
    config_callbacks = config["callbacks"]
    if config_callbacks["dirname"] is None:
        config_callbacks["dirname"] = os.path.abspath(
            os.path.join(dirname(dirname(dirname(__file__))), "experiments"))

        folder_name = config["pipeline"]["name"] + time.strftime(
            "%Y-%m-%d", time.localtime()) + "".join(
                [random.choice(string.ascii_lowercase) for i in range(8)])
        config_callbacks["folder_name"] = os.path.join(
            config_callbacks["dirname"], folder_name)
    else:
        config_callbacks["folder_name"] = config_callbacks["dirname"]

    config_callbacks["checkpoint_dir"] = config_callbacks.get(
        "checkpoint_dir",
        os.path.join(config_callbacks["folder_name"], "checkpoints"))

    config_callbacks["tensorboard_log_dir"] = config_callbacks.get(
        "tensorboard_log_dir",
        os.path.join(config_callbacks["folder_name"], "logs"))


def parse_user_config(user_config_path):
    with open(user_config_path, "r") as f:
        config_user = yaml.load(f)
    dict_merge(config_base, config_user)
    post_process_config(config_base)
    config_save_dir = os.path.join(
        make_valid_dir(config_base["callbacks"]["folder_name"]), "config.yaml")
    with open(config_save_dir, "w") as f:
        yaml.dump(config_base, f)
    return config_base
