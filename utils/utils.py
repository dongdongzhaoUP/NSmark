import gc
import logging
import os
import random
import sys
from pprint import pprint

import numpy as np
import torch
import yaml


def set_logging(save_dir):
    log_format = "%(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)

    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "log.txt")
    if os.path.exists(log_file):
        os.remove(log_file)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    if (len(logger.handlers)) > 1:
        logger.handlers.pop(-1)
    logger.addHandler(fh)


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def cleanup_dataset(dataset):
    """
    Function to clean up memory used by a dataset.

    :param dataset: Dataset object to clear
    """
    del dataset
    gc.collect()


def unload_model(model):
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def load_and_print_config(config_path):
    def load_yaml_file(file_path):
        with open(file_path, "r") as file:
            return list(yaml.safe_load_all(file))

    def print_yaml_dump(data):
        print("\nUsing yaml.dump:")
        print(yaml.dump(data, default_flow_style=False, sort_keys=False))

    def print_pprint(data):
        print("\nUsing pprint for a more readable format:")
        pprint(data, width=1, indent=2)

    try:
        configs = load_yaml_file(config_path)

        print(f"Configuration loaded successfully from {config_path}")
        print(f"Number of documents in the YAML file: {len(configs)}")

        for i, config in enumerate(configs, 1):
            print(f"\n--- Document {i} ---")

            print_pprint(config)

    except yaml.YAMLError as e:
        print(f"Error parsing the YAML file: {e}")
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return None
