"""
utils/logger.py
================
TensorBoard logging wrapper.
Logs each reward component SEPARATELY — critical for debugging reward hacking.
If any single component dominates by 10x others, weights need retuning.
"""

import os
from typing import Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Logger:
    """
    Wraps TensorBoard SummaryWriter with structured logging for MARL training.

    Usage:
        logger = Logger(log_dir="runs/phase1_exp1")
        logger.log_rewards(r1, r2, r_shared, step=global_step)
        logger.log_training(actor1_loss, actor2_loss, critic_loss, step=global_step)
    """

    def __init__(self, log_dir: str, enabled: bool = True):
        self.enabled = enabled and TENSORBOARD_AVAILABLE
        if self.enabled:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logging to: {log_dir}")
            print(f"  Run: tensorboard --logdir {log_dir}")
        else:
            self.writer = None
            if not TENSORBOARD_AVAILABLE:
                print("TensorBoard not available — logging disabled")


    def log_rewards(
        self,
        r1:       float,
        r2:       float,
        r_shared: float,
        step:     int,
    ) -> None:
        """Log reward components separately — key for debugging."""
        if not self.enabled:
            return
        self.writer.add_scalar("rewards/agent1_r1",    r1,       step)
        self.writer.add_scalar("rewards/agent2_r2",    r2,       step)
        self.writer.add_scalar("rewards/shared",       r_shared, step)
        self.writer.add_scalar("rewards/total",        r1 + r2 + r_shared, step)


    def log_episode(
        self,
        episode:          int,
        episode_return1:  float,
        episode_return2:  float,
        episode_length:   int,
        n_failures:       int,
        weighted_tard:    float,
        n_jobs_completed: int,
        avg_health:       float,
    ) -> None:
        """Log per-episode summary metrics."""
        if not self.enabled:
            return
        self.writer.add_scalar("episode/return_agent1",    episode_return1,  episode)
        self.writer.add_scalar("episode/return_agent2",    episode_return2,  episode)
        self.writer.add_scalar("episode/length",           episode_length,   episode)
        self.writer.add_scalar("episode/failures",         n_failures,       episode)
        self.writer.add_scalar("episode/weighted_tardiness", weighted_tard,  episode)
        self.writer.add_scalar("episode/jobs_completed",   n_jobs_completed, episode)
        self.writer.add_scalar("episode/avg_machine_health", avg_health,     episode)


    def log_training(
        self,
        actor1_loss:  float,
        actor2_loss:  float,
        critic_loss:  float,
        entropy1:     float,
        entropy2:     float,
        step:         int,
    ) -> None:
        """Log PPO training losses."""
        if not self.enabled:
            return
        self.writer.add_scalar("train/actor1_loss",  actor1_loss, step)
        self.writer.add_scalar("train/actor2_loss",  actor2_loss, step)
        self.writer.add_scalar("train/critic_loss",  critic_loss, step)
        self.writer.add_scalar("train/entropy1",     entropy1,    step)
        self.writer.add_scalar("train/entropy2",     entropy2,    step)


    def log_scalars(self, tag_value_dict: Dict[str, float], step: int) -> None:
        """Generic scalar logging."""
        if not self.enabled:
            return
        for tag, value in tag_value_dict.items():
            self.writer.add_scalar(tag, value, step)


    def close(self) -> None:
        if self.enabled and self.writer:
            self.writer.close()
