# train.py
""" Run this script for training.

Copyright (c) 2025 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php

USAGE:
    $ python train.py {YOUR CONFIG NAME}.json
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from mymodules.trainers import trainer_provider


parser = argparse.ArgumentParser()
parser.add_argument("config_name", type=str, help="the config name you want to use.")

if __name__ == "__main__":
    args = parser.parse_args()
    config_name = args.config_name

    config_path = Path("configs").joinpath(config_name)

    # Optional, opt-in seed. Upstream NeISF++ has none; when "seed" is present
    # in the config we fix the Python/NumPy/PyTorch RNGs before any model or
    # DataLoader is created, so initialisation and shuffle order are repeatable.
    # This is not a bitwise-determinism guarantee (CUDA kernels may still vary).
    with open(config_path, "r") as f:
        configs = json.load(f)
    seed = configs.get("seed", None)
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"[seed] {seed}")

    trainer = trainer_provider(config_path, is_train=True)
    trainer.train()
