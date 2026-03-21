"""
utils/seeding.py
================
Single entry point to seed ALL sources of randomness in the project.

Why this matters:
    - numpy, torch, and Python's random module each have independent RNG states
    - PettingZoo envs also need a seed
    - Without seeding all of them, results are not reproducible across runs
    - Call seed_everything(seed) at the very start of train.py and evaluate.py

Usage:
    from utils.seeding import seed_everything
    seed_everything(42)
"""

import random
import os
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Seeds numpy, torch (CPU + CUDA), Python stdlib random, and sets
    PYTHONHASHSEED so dict ordering is deterministic.

    Args:
        seed: Integer seed value. Use same seed for reproducible training runs.
              Different seeds for independent experimental runs.
    """
    # Python stdlib random (used in some PettingZoo internals)
    random.seed(seed)

    # Controls hash randomisation for strings/bytes — affects dict ordering
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy — used in all distribution samplers in distributions.py
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU (not needed for 3060 but safe)

    # Makes CUDA ops deterministic — small performance hit, worth it for research
    # NOTE: some CUDA ops have no deterministic implementation; they will raise an error
    # if this is True. If you hit that, set to False and document it.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # disables auto-tuner which introduces non-determinism


def make_env_seed(base_seed: int, env_idx: int) -> int:
    """
    Generates a unique seed for each parallel environment instance.
    Ensures parallel envs don't all sample identical random sequences.

    Args:
        base_seed: Master seed from config
        env_idx:   Environment index (0, 1, 2, 3 for 4 parallel envs)

    Returns:
        Unique integer seed for that env

    Example:
        seeds = [make_env_seed(42, i) for i in range(4)]
        # => [42, 142, 242, 342]
    """
    return base_seed + (env_idx * 100)
