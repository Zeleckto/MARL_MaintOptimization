"""
rewards/reward_fn.py
=====================
Orchestrator: assembles r1, r2, R_shared from component functions.
Called once per full timestep from mfg_env.py after physics resolution.

Loads weights from reward_weights.yaml at init time.
Never hardcodes any coefficient.
"""

import os
import yaml
from typing import List, Tuple, Optional
import numpy as np

from environments.transitions.degradation import MachineState
from environments.transitions.job_dynamics import Job
from rewards.components.shared_reward import compute_shared_reward
from rewards.components.maintenance_reward import compute_maintenance_reward
from rewards.components.scheduling_reward import compute_scheduling_reward


class RewardFunction:
    """
    Centralised reward computation for both agents.

    Usage:
        rf = RewardFunction(config)
        r1, r2, r_shared = rf.compute(...)
    """

    def __init__(self, config: dict):
        # Load weights from yaml file (takes priority over config dict)
        weights_path = os.path.join(
            os.path.dirname(__file__), "reward_weights.yaml"
        )
        if os.path.exists(weights_path):
            with open(weights_path) as f:
                self.weights = yaml.safe_load(f)
        else:
            # Fallback to config
            self.weights = config.get("reward", {})

        self.t_max = config.get("episode", {}).get("t_max_train", 200)


    def compute(
        self,
        maintenance_actions:      List[int],
        ordering_cost:            float,
        machine_states:           List[MachineState],
        newly_failed_machine_ids: List[int],
        jobs:                     List[Job],
        completed_job_ids:        List[int],
        assignment:               Optional[Tuple[int, int, int]],
    ) -> Tuple[float, float, float]:
        """
        Computes all reward components for one timestep.

        Args:
            maintenance_actions:      Agent 1 maintenance actions [n_machines]
            ordering_cost:            Resource ordering cost from resource_dynamics
            machine_states:           Post-tick machine states
            newly_failed_machine_ids: Machines that failed this step
            jobs:                     All active jobs
            completed_job_ids:        Jobs completed this step
            assignment:               Agent 2 action (j,k,m) or None

        Returns:
            (r1, r2, r_shared) — individual rewards for logging
            Agent 1 receives: r1 (already includes lambda*r_shared)
            Agent 2 receives: r2 (already includes lambda*r_shared)
        """
        # Shared failure penalty
        r_shared = compute_shared_reward(
            newly_failed_machine_ids,
            self.weights.get("c_fail", 30.0),
        )

        # Agent 1 reward
        r1 = compute_maintenance_reward(
            maintenance_actions=maintenance_actions,
            ordering_cost=ordering_cost,
            machine_states=machine_states,
            shared_reward=r_shared,
            weights=self.weights,
        )

        # Agent 2 reward
        r2 = compute_scheduling_reward(
            jobs=jobs,
            completed_job_ids=completed_job_ids,
            assignment=assignment,
            machine_states=machine_states,
            shared_reward=r_shared,
            t_max=self.t_max,
            weights=self.weights,
        )

        return r1, r2, r_shared
