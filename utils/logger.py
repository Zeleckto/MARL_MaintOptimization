"""
utils/logger.py
================
TensorBoard logging with all 23 KPIs for Phase 3.

Each reward component logged SEPARATELY — critical for debugging.
If any single component dominates by 10x, weights need retuning.

23 KPI tags:
  rewards/r1, rewards/r2, rewards/r_shared, rewards/total
  episode/return1, episode/return2, episode/length
  episode/failures, episode/n_PM, episode/n_CM
  episode/weighted_tardiness, episode/jobs_completed, episode/jobs_late
  episode/avg_health, episode/avg_hazard_rate
  episode/MTBF, episode/service_level, episode/avg_inventory
  episode/pm_cm_ratio
  train/actor1_loss, train/actor2_loss, train/critic_loss
  train/entropy1, train/entropy2
"""

import os
from typing import Dict, List, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Logger:
    """
    Wraps TensorBoard SummaryWriter with structured logging.

    Usage:
        logger = Logger("runs/phase1_exp1")
        logger.log_rewards(r1, r2, r_shared, step)
        logger.log_episode(metrics_dict, episode)
        logger.log_training(losses_dict, step)
    """

    def __init__(self, log_dir: str, enabled: bool = True):
        self.enabled = enabled and TENSORBOARD_AVAILABLE
        if self.enabled:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard → {log_dir}")
            print(f"  tensorboard --logdir {log_dir}")
        else:
            self.writer = None
            if not TENSORBOARD_AVAILABLE:
                print("TensorBoard not available — logging disabled")

    # ------------------------------------------------------------------
    # Per-step reward logging
    # ------------------------------------------------------------------
    def log_rewards(self, r1: float, r2: float, r_shared: float, step: int) -> None:
        if not self.enabled:
            return
        self.writer.add_scalar("rewards/r1",     r1,       step)
        self.writer.add_scalar("rewards/r2",     r2,       step)
        self.writer.add_scalar("rewards/shared", r_shared, step)
        self.writer.add_scalar("rewards/total",  r1 + r2 + r_shared, step)

    # ------------------------------------------------------------------
    # Per-episode KPI logging (23 tags)
    # ------------------------------------------------------------------
    def log_episode(
        self,
        episode:           int,
        episode_return1:   float,
        episode_return2:   float,
        episode_length:    int,
        n_failures:        int,
        n_PM:              int,
        n_CM:              int,
        weighted_tard:     float,
        n_jobs_completed:  int,
        n_jobs_late:       int,
        avg_health:        float,
        avg_hazard_rate:   float,
        mtbf:              float,
        service_level:     float,       # fraction of jobs on time
        avg_inventory:     float,       # avg consumable inventory
    ) -> None:
        """Logs all 23 per-episode KPIs."""
        if not self.enabled:
            return

        self.writer.add_scalar("episode/return1",           episode_return1,  episode)
        self.writer.add_scalar("episode/return2",           episode_return2,  episode)
        self.writer.add_scalar("episode/length",            episode_length,   episode)
        self.writer.add_scalar("episode/failures",          n_failures,       episode)
        self.writer.add_scalar("episode/n_PM",              n_PM,             episode)
        self.writer.add_scalar("episode/n_CM",              n_CM,             episode)
        self.writer.add_scalar("episode/weighted_tardiness", weighted_tard,   episode)
        self.writer.add_scalar("episode/jobs_completed",    n_jobs_completed, episode)
        self.writer.add_scalar("episode/jobs_late",         n_jobs_late,      episode)
        self.writer.add_scalar("episode/avg_health",        avg_health,       episode)
        self.writer.add_scalar("episode/avg_hazard_rate",   avg_hazard_rate,  episode)
        self.writer.add_scalar("episode/MTBF",              mtbf,             episode)
        self.writer.add_scalar("episode/service_level",     service_level,    episode)
        self.writer.add_scalar("episode/avg_inventory",     avg_inventory,    episode)

        pm_cm_ratio = n_PM / max(n_CM, 1)
        self.writer.add_scalar("episode/pm_cm_ratio",       pm_cm_ratio,      episode)

    # ------------------------------------------------------------------
    # Training loss logging
    # ------------------------------------------------------------------
    def log_training(
        self,
        actor1_loss: float,
        actor2_loss: float,
        critic_loss: float,
        entropy1:    float,
        entropy2:    float,
        step:        int,
    ) -> None:
        if not self.enabled:
            return
        self.writer.add_scalar("train/actor1_loss", actor1_loss, step)
        self.writer.add_scalar("train/actor2_loss", actor2_loss, step)
        self.writer.add_scalar("train/critic_loss", critic_loss, step)
        self.writer.add_scalar("train/entropy1",    entropy1,    step)
        self.writer.add_scalar("train/entropy2",    entropy2,    step)

    # ------------------------------------------------------------------
    # Generic
    # ------------------------------------------------------------------
    def log_scalars(self, tag_value_dict: Dict[str, float], step: int) -> None:
        if not self.enabled:
            return
        for tag, val in tag_value_dict.items():
            self.writer.add_scalar(tag, val, step)

    def close(self) -> None:
        if self.enabled and self.writer:
            self.writer.close()
