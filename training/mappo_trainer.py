from __future__ import annotations
"""
training/mappo_trainer.py
==========================
Main MARL training loop.

Sequence per training iteration:
    1. Collect T rollout steps across N parallel envs (CPU)
    2. Compute GAE advantages for both agents
    3. K epochs of PPO updates (GPU)
    4. Log metrics to TensorBoard
    5. Save checkpoint every N episodes

AEC episode flow:
    For each timestep:
        Agent 1 half-step -> Agent 2 half-step + physics -> rewards

Training phases (stochasticity_level in config):
    Phase 1: Weibull failures only  — validate basic learning
    Phase 2: + LogNormal proc times — add noise
    Phase 3: + Poisson arrivals     — full stochasticity
"""

import os
import time
import numpy as np
from typing import Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from environments.mfg_env import ManufacturingEnv, AGENT_PDM, AGENT_JOBSHOP
from agents.pdm_agent import PDMAgent
from agents.jobshop_agent import JobShopAgent
from models.critic import CentralizedCritic
from training.rollout_buffer import RolloutBuffer
from training.ppo_update import ppo_update, build_optimizers
from utils.logger import Logger
from utils.checkpoint import save_checkpoint
from utils.seeding import seed_everything


class MAPPOTrainer:
    """
    Orchestrates MAPPO training for the manufacturing environment.

    Usage:
        trainer = MAPPOTrainer(config)
        trainer.train()
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if (
            TORCH_AVAILABLE and torch.cuda.is_available()
        ) else "cpu"
        print(f"Training device: {self.device}")

        # Seed everything
        seed_everything(config.get("seed", 42))

        # Initialise environment (single env for simplicity — extend to parallel later)
        self.env = ManufacturingEnv(config)

        # Initialise agents
        self.agent1 = PDMAgent(config, device=self.device)
        self.agent2 = JobShopAgent(config, device=self.device)

        # Initialise critic
        if TORCH_AVAILABLE:
            self.critic = CentralizedCritic(config).to(self.device)
        else:
            self.critic = None

        # Optimizers
        self.optim1, self.optim2, self.optim_critic = build_optimizers(
            self.agent1, self.agent2, self.critic, config
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(config)

        # Logger
        log_dir = config.get("logging", {}).get("tensorboard_dir", "runs/")
        phase   = config.get("stochasticity_level", 1)
        run_name = f"phase{phase}_{int(time.time())}"
        self.logger = Logger(os.path.join(log_dir, run_name))

        # Training config
        mappo       = config.get("mappo", {})
        self.t_max  = config.get("episode", {}).get("t_max_train", 200)
        self.rollout_steps = mappo.get("rollout_steps", 2048)
        self.gamma         = mappo.get("gamma", 0.99)
        self.gae_lambda    = mappo.get("gae_lambda", 0.95)

        # Checkpoint config
        self.ckpt_dir = config.get("logging", {}).get("checkpoint_dir", "checkpoints/")
        self.log_every = config.get("logging", {}).get("log_every_n_episodes", 10)

        # Tracking
        self.global_step = 0
        self.episode     = 0


    def train(self, total_timesteps: int = 500_000) -> None:
        """
        Main training loop.

        Args:
            total_timesteps: Total environment steps to train for.
                             500k is a reasonable start for Phase 1.
        """
        print(f"Starting MAPPO training for {total_timesteps:,} timesteps")
        print(f"Stochasticity phase: {self.config.get('stochasticity_level', 1)}")

        obs_dict, info = self.env.reset(seed=self.config.get("seed", 42))
        obs1_list = obs_dict[AGENT_PDM]
        obs2_dict = self.env.observe(AGENT_JOBSHOP)

        while self.global_step < total_timesteps:
            # ── Rollout collection ────────────────────────────────────────
            self.buffer.reset()

            for _ in range(self.rollout_steps):
                # Agent 1 acts
                action1, logp1, ent1 = self.agent1.act(
                    obs_np=obs1_list,
                    machine_states=self.env.machine_states,
                    machine_busy=self.env.machine_busy,
                    resource_state=self.env.resource_state,
                    rho_PM=self.env.rho_PM,
                    rho_CM=self.env.rho_CM,
                )

                # Estimate value for Agent 1 obs
                v1 = self._estimate_value(obs1_list, obs2_dict)

                # Agent 1 half-step
                self.env.step(action1)

                # Get updated obs2 after Agent 1 half-step
                obs2_dict = self.env.observe(AGENT_JOBSHOP)
                valid_pairs = self.env._valid_pairs

                # Agent 2 acts
                semantic2, action2_idx, logp2, ent2 = self.agent2.act(
                    obs=obs2_dict,
                    valid_pairs=valid_pairs,
                )

                v2 = self._estimate_value(obs1_list, obs2_dict)

                # Agent 2 half-step + physics resolution
                self.env.step(action2_idx)

                r1   = self.env.rewards[AGENT_PDM]
                r2   = self.env.rewards[AGENT_JOBSHOP]
                term = self.env.terminations[AGENT_PDM]
                trunc = self.env.truncations[AGENT_PDM]
                done = term or trunc

                # Store transition
                self.buffer.add(
                    obs1=obs1_list.copy(),
                    action1=action1,
                    logp1=logp1,
                    r1=r1,
                    v1=v1,
                    obs2=obs2_dict,
                    action2=action2_idx,
                    logp2=logp2,
                    r2=r2,
                    v2=v2,
                    done=done,
                    truncated=trunc,
                )

                # Log per-step rewards
                self.logger.log_rewards(r1, r2, 0.0, self.global_step)
                self.global_step += 1

                if done:
                    # Log episode summary
                    if self.episode % self.log_every == 0:
                        avg_health = np.mean([
                            s.health for s in self.env.machine_states
                        ])
                        self.logger.log_episode(
                            episode=self.episode,
                            episode_return1=self.buffer.episode_r1,
                            episode_return2=self.buffer.episode_r2,
                            episode_length=self.buffer.episode_steps,
                            n_failures=self.env._episode_failures,
                            weighted_tard=self.env.job_engine.compute_weighted_tardiness(
                                self.env.jobs
                            ),
                            n_jobs_completed=self.env._episode_completions,
                            avg_health=avg_health,
                        )

                    # Reset
                    obs_dict, _ = self.env.reset()
            obs1_list = obs_dict[AGENT_PDM]
                    obs2_dict    = self.env.observe(AGENT_JOBSHOP)
                    self.episode += 1

            # Compute bootstrap value for GAE
            last_v = self._estimate_value(obs1_list, obs2_dict)
            last_trunc = self.env.truncations[AGENT_PDM]
            bootstrap = last_v if last_trunc else 0.0

            self.buffer.compute_gae(
                last_value1=bootstrap,
                last_value2=bootstrap,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )

            # ── PPO update ────────────────────────────────────────────────
            if TORCH_AVAILABLE and self.critic is not None:
                metrics = ppo_update(
                    agent1=self.agent1,
                    agent2=self.agent2,
                    critic=self.critic,
                    buffer1=self.buffer.buffer1,
                    buffer2=self.buffer.buffer2,
                    optim_actor1=self.optim1,
                    optim_actor2=self.optim2,
                    optim_critic=self.optim_critic,
                    config=self.config,
                )
                self.logger.log_training(
                    actor1_loss=metrics.get("actor1_loss", 0),
                    actor2_loss=metrics.get("actor2_loss", 0),
                    critic_loss=metrics.get("critic_loss", 0),
                    entropy1=metrics.get("entropy1", 0),
                    entropy2=metrics.get("entropy2", 0),
                    step=self.global_step,
                )

            # Save checkpoint every 50k steps
            if self.global_step % 50_000 < self.rollout_steps and TORCH_AVAILABLE:
                save_checkpoint(
                    checkpoint_dir=self.ckpt_dir,
                    episode=self.episode,
                    global_step=self.global_step,
                    actor1=self.agent1.policy,
                    actor2=self.agent2.tgin,
                    critic=self.critic,
                    optim_actor1=self.optim1,
                    optim_actor2=self.optim2,
                    optim_critic=self.optim_critic,
                    config=self.config,
                    tag="latest",
                )

        print(f"Training complete. Total steps: {self.global_step:,}")
        self.logger.close()


    def _estimate_value(self, obs1: np.ndarray, obs2: dict) -> float:
        """
        Estimates V(s) using the critic.
        Returns 0.0 if critic not available (numpy-only mode).
        """
        if not TORCH_AVAILABLE or self.critic is None:
            return 0.0

        import torch

        try:
            from models.tgin.graph_builder import GraphBuilder
            builder = GraphBuilder(self.config)
            graph   = builder.build(obs2, device=self.device)
            emb     = self.agent2.tgin(graph)

            obs1_t   = torch.tensor(obs1, dtype=torch.float32).to(self.device)
            res_flat = torch.tensor(
                self.env.resource_state.to_flat_vector(),
                dtype=torch.float32
            ).to(self.device)

            with torch.no_grad():
                v = self.critic(emb, res_flat, obs1_t)
            return float(v.item())
        except Exception:
            return 0.0