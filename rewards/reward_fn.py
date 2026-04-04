"""
rewards/reward_fn.py
======================
Orchestrator: assembles r1, r2, R_shared from component functions.
Loads weights from reward_weights.yaml. Never hardcodes any coefficient.

KEY CHANGE: passes prev_ruls (pre-tick RUL values) to maintenance_reward
for ΔRUL computation.
"""

import os
import yaml
from typing import List, Optional, Tuple

from environments.transitions.degradation import MachineState, estimate_rul
from environments.transitions.job_dynamics import Job
from rewards.components.shared_reward import compute_shared_reward
from rewards.components.maintenance_reward import compute_maintenance_reward
from rewards.components.scheduling_reward import compute_scheduling_reward


class RewardFunction:
    """
    Centralised reward computation for both agents.

    Usage:
        rf = RewardFunction(config)
        # Before physics tick: save prev_ruls
        prev_ruls = rf.snapshot_ruls(machine_states)
        # ... tick physics ...
        r1, r2, r_shared = rf.compute(prev_ruls=prev_ruls, ...)
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

        self.t_max = config.get("episode", {}).get("t_max_train", 150)

    def snapshot_ruls(self, machine_states: List[MachineState]) -> List[float]:
        """
        Call BEFORE the physics tick to capture RUL snapshot.
        Pass the result as prev_ruls to compute().
        """
        return [estimate_rul(s) for s in machine_states]

    def compute(
        self,
        maintenance_actions:       List[int],
        ordering_cost:             float,
        machine_states:            List[MachineState],
        newly_failed_machine_ids:  List[int],
        jobs:                      List[Job],
        completed_job_ids:         List[int],
        assignment:                Optional[Tuple[int, int, int]],
        current_time:              float,
        prev_ruls:                 Optional[List[float]] = None,
        consumable_inventory:      Optional[List[float]] = None,
    ) -> Tuple[float, float, float]:
        """
        Computes all reward components for one timestep.

        Returns:
            (r1, r2, r_shared)
        """
        # Shared failure penalty
        r_shared = compute_shared_reward(
            newly_failed_machine_ids,
            self.weights.get("c_fail", 25.0),
        )

        # Agent 1 reward (with ΔRUL)
        r1 = compute_maintenance_reward(
            maintenance_actions  = maintenance_actions,
            ordering_cost        = ordering_cost,
            machine_states       = machine_states,
            prev_ruls            = prev_ruls,
            consumable_inventory = consumable_inventory,
            shared_reward        = r_shared,
            weights              = self.weights,
        )

        # Agent 2 reward
        r2 = compute_scheduling_reward(
            jobs              = jobs,
            completed_job_ids = completed_job_ids,
            assignment        = assignment,
            machine_states    = machine_states,
            shared_reward     = r_shared,
            current_time      = current_time,
            t_max             = self.t_max,
            weights           = self.weights,
        )

        return r1, r2, r_shared
