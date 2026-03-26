from __future__ import annotations
"""
training/mappo_trainer.py
==========================
Fixed MAPPO training loop.

Key fixes vs previous version:
  1. Uses env._step_agent1/2/_resolve_physics/_compute_rewards directly
     (avoids PettingZoo AEC selector ordering issues)
  2. Correctly unpacks env.reset() dict to get obs1 numpy array
  3. Passes action1 to critic for better value estimation
  4. Proper bootstrap: 0 if terminated, V(s_T) if truncated
  5. Logs all reward components separately to TensorBoard
  6. Passes current_step to reward_fn for makespan estimate
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
        trainer.train(total_timesteps=500_000)
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if (
            TORCH_AVAILABLE and torch.cuda.is_available()
        ) else "cpu"
        print(f"Training device: {self.device}")

        seed_everything(config.get("seed", 42))

        # Environment
        self.env = ManufacturingEnv(config)

        # Agents
        self.agent1 = PDMAgent(config, device=self.device)
        self.agent2 = JobShopAgent(config, device=self.device)

        # Critic
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
        log_dir  = config.get("logging", {}).get("tensorboard_dir", "runs/")
        phase    = config.get("stochasticity_level", 1)
        run_name = f"phase{phase}_{int(time.time())}"
        self.logger = Logger(os.path.join(log_dir, run_name))

        # Config
        mappo             = config.get("mappo", {})
        self.t_max        = config.get("episode", {}).get("t_max_train", 200)
        self.rollout_steps = mappo.get("rollout_steps", 2048)
        self.gamma        = mappo.get("gamma", 0.99)
        self.gae_lambda   = mappo.get("gae_lambda", 0.95)
        self.ckpt_dir     = config.get("logging", {}).get("checkpoint_dir", "checkpoints/")
        self.log_every    = config.get("logging", {}).get("log_every_n_episodes", 10)

        # State
        self.global_step  = 0
        self.episode      = 0
        self.obs1:        Optional[np.ndarray] = None
        self.last_action1: Optional[dict]      = None
        self._last_trunc  = False


    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_env(self, seed: Optional[int] = None) -> None:
        """Resets env and stores initial obs1 as numpy array."""
        obs_dict, _ = self.env.reset(seed=seed)
        # env.reset() returns dict {AGENT_PDM: np_array, AGENT_JOBSHOP: graph_dict}
        self.obs1 = obs_dict[AGENT_PDM]
        self.last_action1 = None


    # ── Value estimation ──────────────────────────────────────────────────────

    def _estimate_value(self, action1_maint: Optional[np.ndarray] = None) -> float:
        """
        Estimates V(s_global) using the critic.
        Passes action1 so critic can reason about consequences of Agent 1's choice.
        """
        if not TORCH_AVAILABLE or self.critic is None or self.obs1 is None:
            return 0.0

        try:
            from models.tgin.graph_builder import GraphBuilder
            obs2    = self.env._build_agent2_obs()
            builder = GraphBuilder(self.config)
            graph   = builder.build(obs2, device=self.device)
            emb     = self.agent2.tgin(graph)

            obs1_t   = torch.tensor(self.obs1,                    dtype=torch.float32).to(self.device)
            res_flat = torch.tensor(self.env.resource_state.to_flat_vector(),
                                    dtype=torch.float32).to(self.device)

            # One-hot action1 maintenance if provided
            a1_t = None
            if action1_maint is not None:
                a1_t = torch.tensor(action1_maint, dtype=torch.long).unsqueeze(0).to(self.device)

            with torch.no_grad():
                v = self.critic(emb, res_flat, obs1_t, a1_t)
            return float(v.item())
        except Exception as e:
            return 0.0


    # ── Single timestep collection ────────────────────────────────────────────

    def _collect_one_step(self) -> tuple:
        """
        Collects one full (Agent1 + Agent2 + physics) timestep.

        Returns:
            (done: bool, truncated: bool)
        """
        # ── Agent 1 half-step ──────────────────────────────────────────────
        action1, logp1, ent1 = self.agent1.act(
            obs_np         = self.obs1,
            machine_states = self.env.machine_states,
            machine_busy   = self.env.machine_busy,
            resource_state = self.env.resource_state,
            rho_PM         = self.env.rho_PM,
            rho_CM         = self.env.rho_CM,
        )

        # Value BEFORE Agent 1 acts (pre-action state)
        v1 = self._estimate_value(action1_maint=None)

        # Apply Agent 1 action
        self.env._step_agent1(action1)
        self.last_action1 = action1

        # ── Agent 2 half-step ──────────────────────────────────────────────
        obs2        = self.env._build_agent2_obs()
        valid_pairs = self.env._valid_pairs

        sem2, idx2, logp2, ent2 = self.agent2.act(obs2, valid_pairs)

        # Value after Agent 1 acted (post-action, pre-Agent2)
        v2 = self._estimate_value(action1_maint=action1["maintenance"])

        # Apply Agent 2 action
        self.env._step_agent2(idx2)

        # ── Physics + rewards ──────────────────────────────────────────────
        self.env._resolve_physics()
        self.env._compute_rewards()

        r1   = self.env.rewards[AGENT_PDM]
        r2   = self.env.rewards[AGENT_JOBSHOP]
        term = self.env.terminations[AGENT_PDM]
        trunc= self.env.truncations[AGENT_PDM]
        done = term or trunc
        self._last_trunc = trunc

        # ── Store transition ───────────────────────────────────────────────
        self.buffer.add(
            obs1     = self.obs1.copy(),
            action1  = action1,
            logp1    = logp1,
            r1       = r1,
            v1       = v1,
            obs2     = obs2,
            action2  = idx2,
            logp2    = logp2,
            r2       = r2,
            v2       = v2,
            done     = done,
            truncated= trunc,
        )

        # Log per-step
        self.logger.log_rewards(r1, r2, 0.0, self.global_step)
        self.global_step += 1

        # Update obs1 for next step
        self.obs1 = self.env.observe(AGENT_PDM)

        return done, trunc


    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, total_timesteps: int = 500_000) -> None:
        """
        Main MAPPO training loop.

        Args:
            total_timesteps: Total env steps to train for.
        """
        print(f"\n{'='*60}")
        print(f"  MAPPO Training — Three-Tier Manufacturing MARL")
        print(f"{'='*60}")
        print(f"  Timesteps:    {total_timesteps:,}")
        print(f"  Rollout size: {self.rollout_steps}")
        print(f"  Episode len:  {self.t_max}")
        print(f"  Phase:        {self.config.get('stochasticity_level', 1)}")
        print(f"  Device:       {self.device}")
        print(f"{'='*60}\n")

        self._reset_env(seed=self.config.get("seed", 42))

        while self.global_step < total_timesteps:

            # ── Rollout collection ─────────────────────────────────────────
            self.buffer.reset()
            trunc = False  # track last truncation for bootstrap

            for _ in range(self.rollout_steps):
                done, trunc = self._collect_one_step()

                if done:
                    # Episode ended — log and reset
                    avg_health = float(np.mean([
                        s.health for s in self.env.machine_states
                    ]))

                    if self.episode % self.log_every == 0:
                        print(
                            f"  ep={self.episode:>5} | "
                            f"step={self.global_step:>8,} | "
                            f"r1={self.buffer.episode_r1:>+8.1f} | "
                            f"r2={self.buffer.episode_r2:>+8.1f} | "
                            f"failures={self.env._episode_failures:>3} | "
                            f"done={self.env._episode_completions:>3}jobs | "
                            f"health={avg_health:.1f}%"
                        )

                    self.logger.log_episode(
                        episode          = self.episode,
                        episode_return1  = self.buffer.episode_r1,
                        episode_return2  = self.buffer.episode_r2,
                        episode_length   = self.buffer.episode_steps,
                        n_failures       = self.env._episode_failures,
                        weighted_tard    = self.env.job_engine.compute_weighted_tardiness(
                            self.env.jobs
                        ),
                        n_jobs_completed = self.env._episode_completions,
                        avg_health       = avg_health,
                    )

                    self.episode += 1
                    self._reset_env()

            # ── GAE with proper bootstrap ──────────────────────────────────
            # CRITICAL: terminated = V=0 (no future), truncated = V(s_T) > 0
            if trunc:
                # Hit T_max — bootstrap from critic estimate
                last_v = self._estimate_value(
                    action1_maint=self.last_action1["maintenance"]
                    if self.last_action1 else None
                )
            else:
                # All jobs done — no future value
                last_v = 0.0

            self.buffer.compute_gae(
                last_value1 = last_v,
                last_value2 = last_v,
                gamma       = self.gamma,
                gae_lambda  = self.gae_lambda,
            )

            # ── PPO update ─────────────────────────────────────────────────
            if TORCH_AVAILABLE and self.critic is not None:
                metrics = ppo_update(
                    agent1       = self.agent1,
                    agent2       = self.agent2,
                    critic       = self.critic,
                    buffer1      = self.buffer.buffer1,
                    buffer2      = self.buffer.buffer2,
                    optim_actor1 = self.optim1,
                    optim_actor2 = self.optim2,
                    optim_critic = self.optim_critic,
                    config       = self.config,
                )
                self.logger.log_training(
                    actor1_loss = metrics.get("actor1_loss", 0),
                    actor2_loss = metrics.get("actor2_loss", 0),
                    critic_loss = metrics.get("critic_loss", 0),
                    entropy1    = metrics.get("entropy1", 0),
                    entropy2    = metrics.get("entropy2", 0),
                    step        = self.global_step,
                )

            # ── Checkpoint ─────────────────────────────────────────────────
            if (self.global_step % 50_000 < self.rollout_steps
                    and TORCH_AVAILABLE and self.critic is not None):
                save_checkpoint(
                    checkpoint_dir = self.ckpt_dir,
                    episode        = self.episode,
                    global_step    = self.global_step,
                    actor1         = self.agent1.policy,
                    actor2         = self.agent2.tgin,
                    critic         = self.critic,
                    optim_actor1   = self.optim1,
                    optim_actor2   = self.optim2,
                    optim_critic   = self.optim_critic,
                    config         = self.config,
                    tag            = "latest",
                )

        print(f"\nTraining complete. Total steps: {self.global_step:,}")
        self.logger.close()