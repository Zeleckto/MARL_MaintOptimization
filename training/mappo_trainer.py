"""
training/mappo_trainer.py
==========================
Main MAPPO training loop.

Sequence per training iteration:
    1. Collect rollout_steps transitions across n_envs parallel envs
    2. Compute GAE advantages (truncation vs termination handled correctly)
    3. Run K epochs of PPO updates on both actors + critic
    4. Log to TensorBoard
    5. Save checkpoint every N episodes

Usage:
    trainer = MAPPOTrainer(config)
    trainer.train(total_steps=1_000_000)
"""

import os
import yaml
import numpy as np
from typing import Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from environments.mfg_env import ManufacturingEnv
from agents.pdm_agent import PDMAgent
from agents.jobshop_agent import JobShopAgent
from models.critic import CentralizedCritic
from training.rollout_buffer import RolloutBuffer
from training.ppo_update import PPOUpdater
from utils.logger import Logger
from utils.checkpoint import save_checkpoint
from utils.seeding import seed_everything


class MAPPOTrainer:
    """
    Full MAPPO training pipeline for the manufacturing environment.
    Runs single-env training (no parallel envs) for simplicity.
    Extend to VecManufacturingEnv for parallel rollout collection.
    """

    def __init__(self, config: dict):
        self.config = config
        mappo = config.get("mappo", {})

        # Seeding
        seed = config.get("seed", 42)
        seed_everything(seed)

        # Device
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"Training on: {self.device}")

        # Environment
        self.env = ManufacturingEnv(config)

        # Agents
        self.pdm_agent      = PDMAgent(config, self.device)
        self.jobshop_agent  = JobShopAgent(config, self.device)

        # Critic
        if TORCH_AVAILABLE:
            self.critic = CentralizedCritic(config).to(self.device)
        else:
            self.critic = None

        # Rollout buffer
        self.buffer = RolloutBuffer(config)

        # PPO updater
        if TORCH_AVAILABLE:
            self.updater = PPOUpdater(
                config,
                self.pdm_agent.policy,
                self.jobshop_agent,
                self.critic,
                self.device,
            )
        else:
            self.updater = None

        # Logger
        log_dir = config.get("logging", {}).get("tensorboard_dir", "runs/")
        self.logger = Logger(log_dir=os.path.join(log_dir, "phase1_run1"))

        # Training config
        self.rollout_steps   = mappo.get("rollout_steps",  2048)
        self.t_max           = config.get("episode", {}).get("t_max_train", 200)
        self.checkpoint_dir  = config.get("logging", {}).get("checkpoint_dir", "checkpoints/")
        self.log_every       = config.get("logging", {}).get("log_every_n_episodes", 10)
        self.gamma           = mappo.get("gamma",      0.99)
        self.gae_lambda      = mappo.get("gae_lambda", 0.95)

        # Tracking
        self.global_step   = 0
        self.episode       = 0
        self.best_return   = -float("inf")


    def train(self, total_steps: int = 1_000_000) -> None:
        """
        Main training loop. Runs until total_steps environment steps collected.

        Args:
            total_steps: Total environment steps to train for
        """
        print(f"Starting MAPPO training for {total_steps:,} steps")
        print(f"  Rollout steps per update: {self.rollout_steps}")
        print(f"  Episode length: {self.t_max}")
        print()

        # Reset environment
        obs_dict, _ = self.env.reset(seed=self.config.get("seed", 42))
        obs1 = obs_dict["pdm_agent"]
        obs2 = obs_dict["jobshop_agent"]

        self.buffer.reset()
        ep_r1, ep_r2 = 0.0, 0.0

        while self.global_step < total_steps:

            # ─── Collect rollout ───────────────────────────────────────────
            for _ in range(self.rollout_steps):

                # Agent 1 acts
                action1, logp1, ent1 = self.pdm_agent.act(
                    obs_np=obs1,
                    machine_states=self.env.machine_states,
                    machine_busy=self.env.machine_busy,
                    resource_state=self.env.resource_state,
                    rho_PM=self.env.rho_PM,
                    rho_CM=self.env.rho_CM,
                )

                # Get critic value estimate
                v1 = self._get_value(obs1, obs2)

                # Agent 1 half-step
                self.env.agent_selection = "pdm_agent"
                self.env.step(action1)

                # Agent 2 acts (post Agent-1 state)
                obs2_updated = self.env._build_agent2_obs()
                semantic_action, action2_idx, logp2, ent2 = self.jobshop_agent.act(
                    obs=obs2_updated,
                    valid_pairs=self.env._valid_pairs,
                )
                v2 = v1  # same global state value (approximation)

                # Agent 2 half-step + physics resolution
                self.env.agent_selection = "jobshop_agent"
                self.env.step(action2_idx)

                # Get rewards and done flags
                r1   = self.env.rewards["pdm_agent"]
                r2   = self.env.rewards["jobshop_agent"]
                term = self.env.terminations["pdm_agent"]
                trunc = self.env.truncations["pdm_agent"]
                done  = term or trunc

                # Store in buffer
                self.buffer.add(
                    obs1=obs1,     action1=action1, logp1=logp1, r1=r1, v1=v1,
                    obs2=obs2,     action2=action2_idx, logp2=logp2, r2=r2, v2=v2,
                    done=done,     truncated=trunc,
                )

                ep_r1 += r1
                ep_r2 += r2
                self.global_step += 1

                # Log per-step rewards
                self.logger.log_rewards(r1, r2, self.env._last_r_shared if hasattr(self.env, '_last_r_shared') else 0.0, self.global_step)

                # Get next observations
                obs1 = self.env._build_agent1_obs()
                obs2 = self.env._build_agent2_obs()

                # Handle episode end
                if done:
                    self.episode += 1

                    # Log episode stats
                    if self.episode % self.log_every == 0:
                        avg_health = np.mean([s.health for s in self.env.machine_states])
                        self.logger.log_episode(
                            episode=self.episode,
                            episode_return1=ep_r1,
                            episode_return2=ep_r2,
                            episode_length=self.env.current_step,
                            n_failures=self.env._episode_failures,
                            weighted_tard=self.env.job_engine.compute_weighted_tardiness(self.env.jobs),
                            n_jobs_completed=self.env._episode_completions,
                            avg_health=avg_health,
                        )
                        print(
                            f"Ep {self.episode:>5} | "
                            f"Steps {self.global_step:>8,} | "
                            f"R1={ep_r1:>8.2f} R2={ep_r2:>8.2f} | "
                            f"Failures={self.env._episode_failures} | "
                            f"Completed={self.env._episode_completions}"
                        )

                    ep_r1, ep_r2 = 0.0, 0.0
                    obs_dict, _ = self.env.reset()
                    obs1 = obs_dict["pdm_agent"]
                    obs2 = obs_dict["jobshop_agent"]

            # ─── Compute GAE ───────────────────────────────────────────────
            last_v = self._get_value(obs1, obs2)
            # For truncated: bootstrap with critic. For terminated: 0 (handled in GAE)
            self.buffer.compute_gae(
                last_value1=last_v,
                last_value2=last_v,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )

            # ─── PPO update ────────────────────────────────────────────────
            if self.updater:
                losses = self.updater.update(self.buffer)
                self.logger.log_training(
                    actor1_loss=losses.get("actor1_loss", 0),
                    actor2_loss=losses.get("actor2_loss", 0),
                    critic_loss=losses.get("critic_loss", 0),
                    entropy1=losses.get("entropy1", 0),
                    entropy2=losses.get("entropy2", 0),
                    step=self.global_step,
                )

            self.buffer.reset()

            # ─── Checkpoint ────────────────────────────────────────────────
            if TORCH_AVAILABLE and self.episode % 100 == 0 and self.episode > 0:
                save_checkpoint(
                    checkpoint_dir=self.checkpoint_dir,
                    episode=self.episode,
                    global_step=self.global_step,
                    actor1=self.pdm_agent.policy,
                    actor2=self.jobshop_agent.tgin,
                    critic=self.critic,
                    optim_actor1=self.updater.optim_actor1,
                    optim_actor2=self.updater.optim_actor2,
                    optim_critic=self.updater.optim_critic,
                    config=self.config,
                    tag="latest",
                )

        self.logger.close()
        print(f"Training complete. Total steps: {self.global_step:,}")


    def _get_value(self, obs1: np.ndarray, obs2: dict) -> float:
        """Gets critic value estimate for current global state."""
        if not TORCH_AVAILABLE or self.critic is None:
            return 0.0

        import torch
        # Simplified: use zero embeddings as placeholder
        # Full implementation builds graph, runs TGIN, passes to critic
        return 0.0
