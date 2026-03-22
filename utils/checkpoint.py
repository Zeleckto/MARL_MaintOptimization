"""
utils/checkpoint.py
====================
Save and load model checkpoints with full training metadata.
Saves theta1, theta2, phi (both actors + critic) plus optimizer states.
"""

import os
import json
from typing import Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def save_checkpoint(
    checkpoint_dir: str,
    episode:        int,
    global_step:    int,
    actor1,         # nn.Module
    actor2,         # nn.Module
    critic,         # nn.Module
    optim_actor1,
    optim_actor2,
    optim_critic,
    config:         dict,
    tag:            str = "latest",
) -> str:
    """
    Saves full training state to checkpoint_dir/tag.pt

    Args:
        checkpoint_dir: Directory to save checkpoint
        episode:        Current episode number
        global_step:    Total environment steps so far
        actor1/2:       Agent policy networks (nn.Module)
        critic:         Centralized critic (nn.Module)
        optim_*:        Optimizer states
        config:         Config dict (saved as metadata)
        tag:            Filename tag ('latest', 'best', 'ep_500' etc.)

    Returns:
        Path to saved checkpoint file
    """
    if not TORCH_AVAILABLE:
        print("PyTorch not available — checkpoint not saved")
        return ""

    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{tag}.pt")

    torch.save({
        "episode":          episode,
        "global_step":      global_step,
        "actor1_state":     actor1.state_dict(),
        "actor2_state":     actor2.state_dict(),
        "critic_state":     critic.state_dict(),
        "optim_actor1":     optim_actor1.state_dict(),
        "optim_actor2":     optim_actor2.state_dict(),
        "optim_critic":     optim_critic.state_dict(),
        "config":           config,
    }, path)

    print(f"Checkpoint saved: {path} (episode={episode}, step={global_step})")
    return path


def load_checkpoint(
    path:        str,
    actor1,
    actor2,
    critic,
    optim_actor1 = None,
    optim_actor2 = None,
    optim_critic  = None,
    device:      str = "cuda",
) -> dict:
    """
    Loads checkpoint from path into provided model instances.

    Args:
        path:    Path to .pt checkpoint file
        actor1/2, critic: nn.Module instances (modified in place)
        optim_*: Optimizer instances (optional — pass None to skip)
        device:  Device to load tensors onto

    Returns:
        Metadata dict with episode, global_step, config
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for checkpoint loading")

    ckpt = torch.load(path, map_location=device)

    actor1.load_state_dict(ckpt["actor1_state"])
    actor2.load_state_dict(ckpt["actor2_state"])
    critic.load_state_dict(ckpt["critic_state"])

    if optim_actor1 is not None:
        optim_actor1.load_state_dict(ckpt["optim_actor1"])
    if optim_actor2 is not None:
        optim_actor2.load_state_dict(ckpt["optim_actor2"])
    if optim_critic is not None:
        optim_critic.load_state_dict(ckpt["optim_critic"])

    print(f"Checkpoint loaded: {path}")
    print(f"  Resuming from episode={ckpt['episode']}, step={ckpt['global_step']}")

    return {
        "episode":     ckpt["episode"],
        "global_step": ckpt["global_step"],
        "config":      ckpt.get("config", {}),
    }
