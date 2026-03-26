from __future__ import annotations
"""
rewards/reward_fn.py
=====================
Orchestrator: assembles r1, r2, R_shared from all components.
Called once per full timestep from mfg_env._compute_rewards().

Changes:
  - Criticality-weighted R_shared (bottleneck failures cost more)
  - RUL preservation bonus in r1
  - Makespan estimate penalty in r2
  - delta_obj weighting on resource ordering cost
  - Passes eta_values to maintenance_reward for RUL normalisation
"""

import os
import yaml
from typing import List, Tuple, Optional, Dict

from environments.transitions.degradation import MachineState
from environments.transitions.job_dynamics import Job
from rewards.components.shared_reward import (
    compute_shared_reward, compute_machine_criticality
)
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
        weights_path = os.path.join(
            os.path.dirname(__file__), "reward_weights.yaml"
        )
        if os.path.exists(weights_path):
            with open(weights_path) as f:
                self.weights = yaml.safe_load(f)
        else:
            self.weights = config.get("reward", {})

        # Extract eta values per machine for RUL normalisation
        self.eta_values = [
            m.get("eta", 3000.0)
            for m in config.get("machines", [])
        ]

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
        current_step:             int = 0,
        eligible_map:             Optional[Dict[int, List[int]]] = None,
        n_pending_ops:            int = 0,
    ) -> Tuple[float, float, float]:
        """
        Computes all reward components for one timestep.

        Args:
            maintenance_actions:      Agent 1 maintenance actions [n_machines]
            ordering_cost:            Raw resource ordering cost (before δ)
            machine_states:           Post-tick machine states
            newly_failed_machine_ids: Machines that failed this step
            jobs:                     All jobs
            completed_job_ids:        Jobs completed this step
            assignment:               Agent 2 action (j,k,m) or None
            current_step:             Current timestep (for makespan estimate)
            eligible_map:             {machine_id: [ops]} for criticality
            n_pending_ops:            Total pending ops (for criticality)

        Returns:
            (r1, r2, r_shared)
        """
        # Criticality-weighted shared failure penalty
        machine_criticality = compute_machine_criticality(
            newly_failed_machine_ids,
            eligible_map or {},
            n_pending_ops,
        )
        r_shared = compute_shared_reward(
            newly_failed_machine_ids,
            c_fail                 = self.weights.get("c_fail", 30.0),
            criticality_multiplier = self.weights.get("criticality_multiplier", 5.0),
            machine_criticality    = machine_criticality,
        )

        # Agent 1 reward (with RUL bonus and δ-weighted ordering)
        r1 = compute_maintenance_reward(
            maintenance_actions = maintenance_actions,
            ordering_cost       = ordering_cost,
            machine_states      = machine_states,
            eta_values          = self.eta_values,
            shared_reward       = r_shared,
            weights             = self.weights,
        )

        # Agent 2 reward (with makespan estimate)
        r2 = compute_scheduling_reward(
            jobs              = jobs,
            completed_job_ids = completed_job_ids,
            assignment        = assignment,
            machine_states    = machine_states,
            shared_reward     = r_shared,
            t_max             = self.t_max,
            current_step      = current_step,
            weights           = self.weights,
        )

        return r1, r2, r_shared