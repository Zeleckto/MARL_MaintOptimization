"""
training/rollout_buffer.py
===========================
Stores transitions for both agents and computes GAE advantages.

Key design points:
    1. Separate buffers per agent (different obs/action spaces)
    2. GAE computed at episode end
    3. CRITICAL: distinguishes truncation vs termination for bootstrap value
       - Terminated (all jobs done):  bootstrap V = 0
       - Truncated  (hit T_max):      bootstrap V = V_phi(s_T) (critic estimate)
    4. Returns generator for minibatch sampling during PPO update
"""

from typing import List, Tuple, Optional, Generator
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AgentBuffer:
    """
    Rollout buffer for a single agent.
    Stores T transitions then computes GAE.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.obs       = []   # observations
        self.actions   = []   # actions taken
        self.log_probs = []   # log pi(a|s) at time of collection
        self.rewards   = []   # r_t
        self.values    = []   # V(s_t) from critic
        self.dones     = []   # True if episode ended (term OR trunc)
        self.truncated = []   # True if episode TRUNCATED (not terminated)
        self.advantages = []
        self.returns    = []

    def add(
        self,
        obs,
        action,
        log_prob: float,
        reward:   float,
        value:    float,
        done:     bool,
        truncated: bool = False,
    ):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.truncated.append(truncated)

    def compute_gae(
        self,
        last_value: float,  # V(s_{T+1}) — 0 if terminated, V(s_T) if truncated
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        Computes Generalised Advantage Estimates (GAE).

        GAE formula:
            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}

        The last_value argument is the bootstrap value:
            - 0.0 if episode TERMINATED (all jobs done) — no future value
            - V(s_T) if episode TRUNCATED (hit T_max) — critic estimates remaining

        This distinction is MANDATORY for correct advantage estimates.
        Using 0 for truncation underestimates future rewards.
        Using V for termination overestimates (there IS no future).

        Args:
            last_value: Bootstrap value for s_{T+1}
            gamma:      Discount factor
            gae_lambda: GAE lambda parameter
        """
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        # Compute backwards from T to 0
        for t in reversed(range(T)):
            # Next value: last_value if at episode boundary, else V(s_{t+1})
            if t == T - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            # Handle done: if done, next state has no value
            # BUT distinguish terminated vs truncated
            if self.dones[t] and not self.truncated[t]:
                # Terminated: future value is 0
                next_value = 0.0

            # TD error
            delta = self.rewards[t] + gamma * next_value - self.values[t]

            # GAE accumulation
            gae = delta + gamma * gae_lambda * (0.0 if self.dones[t] else gae)
            advantages[t] = gae

        self.advantages = advantages
        self.returns    = advantages + np.array(self.values, dtype=np.float32)

    def get_minibatches(
        self,
        minibatch_size: int,
        shuffle:        bool = True,
    ) -> Generator:
        """
        Generator yielding minibatches for PPO update.
        Yields dicts with keys: obs, actions, log_probs, advantages, returns
        """
        T = len(self.rewards)
        indices = np.arange(T)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, T, minibatch_size):
            end   = start + minibatch_size
            batch_idx = indices[start:end]
            yield {
                "obs":        [self.obs[i] for i in batch_idx],
                "actions":    [self.actions[i] for i in batch_idx],
                "log_probs":  np.array([self.log_probs[i] for i in batch_idx], dtype=np.float32),
                "advantages": self.advantages[batch_idx],
                "returns":    self.returns[batch_idx],
                "values":     np.array([self.values[i] for i in batch_idx], dtype=np.float32),
            }

    def __len__(self):
        return len(self.rewards)


class RolloutBuffer:
    """
    Combined rollout buffer for both agents + critic values.
    """

    def __init__(self, config: dict):
        capacity = config.get("mappo", {}).get("rollout_steps", 2048)
        self.buffer1  = AgentBuffer(capacity)   # Agent 1 (PDM)
        self.buffer2  = AgentBuffer(capacity)   # Agent 2 (Job Shop)
        self.capacity = capacity

        # Shared episode tracking
        self.episode_r1    = 0.0
        self.episode_r2    = 0.0
        self.episode_steps = 0

    def reset(self):
        self.buffer1.reset()
        self.buffer2.reset()
        self.episode_r1    = 0.0
        self.episode_r2    = 0.0
        self.episode_steps = 0

    def add(
        self,
        obs1,   action1,   logp1:   float, r1: float, v1: float,
        obs2,   action2,   logp2:   float, r2: float, v2: float,
        done:   bool,
        truncated: bool = False,
    ):
        self.buffer1.add(obs1, action1, logp1, r1, v1, done, truncated)
        self.buffer2.add(obs2, action2, logp2, r2, v2, done, truncated)
        self.episode_r1    += r1
        self.episode_r2    += r2
        self.episode_steps += 1

    def compute_gae(
        self,
        last_value1: float,
        last_value2: float,
        gamma:       float,
        gae_lambda:  float,
    ):
        self.buffer1.compute_gae(last_value1, gamma, gae_lambda)
        self.buffer2.compute_gae(last_value2, gamma, gae_lambda)

    def is_full(self) -> bool:
        return len(self.buffer1) >= self.capacity
