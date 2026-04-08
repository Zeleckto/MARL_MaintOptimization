from __future__ import annotations
"""
rewards/reward_fn.py
=====================
Orchestrator: assembles r1, r2, R_shared from all components.

Uses inspect.signature to probe each component function at init time,
so this file works with any v0 or v1 version of the components.
If a parameter was removed from a component, we simply don't pass it.
"""

import os
import inspect
import yaml
from typing import List, Tuple, Optional, Dict

import numpy as np

from environments.transitions.degradation import MachineState
from environments.transitions.job_dynamics import Job

from rewards.components.maintenance_reward import compute_maintenance_reward
from rewards.components.scheduling_reward import compute_scheduling_reward

# --- shared_reward: safe conditional imports ---
from rewards.components.shared_reward import compute_shared_reward
try:
    from rewards.components.shared_reward import compute_machine_criticality
except ImportError:
    def compute_machine_criticality(newly_failed, eligible_map, n_pending_ops):
        return {m: 0.0 for m in newly_failed}


class RewardFunction:
    """
    Centralised reward computation for both agents.
    Probes component function signatures at init so calls are always compatible.

    Usage:
        rf = RewardFunction(config)
        r1, r2, r_shared = rf.compute(...)
    """

    def __init__(self, config: dict):
        weights_path = os.path.join(os.path.dirname(__file__), "reward_weights.yaml")
        if os.path.exists(weights_path):
            with open(weights_path) as f:
                self.weights = yaml.safe_load(f)
        else:
            self.weights = config.get("reward", {})

        self.eta_values = [
            m.get("eta", 3000.0) for m in config.get("machines", [])
        ]
        self.t_max = config.get("episode", {}).get("t_max_train", 150)

        # Probe component function signatures once at init.
        # Store sets of accepted param names for each function.
        self._shared_params  = set(inspect.signature(compute_shared_reward).parameters)
        self._maint_params   = set(inspect.signature(compute_maintenance_reward).parameters)
        self._sched_params   = set(inspect.signature(compute_scheduling_reward).parameters)


    def _call_shared(
        self,
        newly_failed:        List[int],
        machine_criticality: Dict[int, float],
        n_completions:       int = 0,
    ) -> float:
        """Calls compute_shared_reward with only the params it accepts."""
        kwargs = {
            "newly_failed_machine_ids": newly_failed,
            "c_fail": self.weights.get("c_fail", 25.0),
        }
        if "criticality_multiplier" in self._shared_params:
            kwargs["criticality_multiplier"] = self.weights.get(
                "criticality_multiplier", 5.0)
        if "machine_criticality" in self._shared_params:
            kwargs["machine_criticality"] = machine_criticality
        if "n_completions" in self._shared_params:
            kwargs["n_completions"] = n_completions
        if "w_comp_shared" in self._shared_params:
            kwargs["w_comp_shared"] = self.weights.get("w_comp_shared", 1.0)
        return compute_shared_reward(**kwargs)


    def _call_maintenance(
        self,
        maintenance_actions: List[int],
        ordering_cost:       float,
        machine_states:      List[MachineState],
        shared_reward:       float,
        inventory_total:     float,
        delta_ruls:          Optional[List[float]] = None,
        n_auto_cm:           int = 0,
        units_ordered:       float = 0.0,
        inv_below_rop:       bool  = False,
        pre_maint_health:    list  = None,
    ) -> float:
        """Calls compute_maintenance_reward with only the params it accepts."""
        kwargs = {
            "maintenance_actions": maintenance_actions,
            "ordering_cost":       ordering_cost,
            "machine_states":      machine_states,
            "shared_reward":       shared_reward,
            "weights":             self.weights,
            "n_auto_cm":           n_auto_cm,
        }
        if "eta_values" in self._maint_params:
            kwargs["eta_values"] = self.eta_values
        if "inventory_total" in self._maint_params:
            kwargs["inventory_total"] = inventory_total
        if "delta_ruls" in self._maint_params and delta_ruls is not None:
            kwargs["delta_ruls"] = delta_ruls
        # n_auto_cm is already in kwargs from the call signature
        if "n_auto_cm" not in self._maint_params:
            kwargs.pop("n_auto_cm", None)
        if "units_ordered" in self._maint_params:
            kwargs["units_ordered"] = units_ordered
        if "inv_below_rop" in self._maint_params:
            kwargs["inv_below_rop"] = inv_below_rop
        if "pre_maint_health" in self._maint_params and pre_maint_health is not None:
            kwargs["pre_maint_health"] = pre_maint_health
        return compute_maintenance_reward(**kwargs)


    def _call_scheduling(
        self,
        jobs:             List[Job],
        completed_ids:    List[int],
        assignment,
        machine_states:   List[MachineState],
        shared_reward:    float,
        current_step:     int,
        t_max:            int = 150,
    ) -> float:
        """Calls compute_scheduling_reward with only the params it accepts.
        Handles both current_step (zip) and current_time (v1) naming.
        """
        kwargs = {
            "jobs":              jobs,
            "completed_job_ids": completed_ids,
            "assignment":        assignment,
            "shared_reward":     shared_reward,
            "weights":           self.weights,
        }
        if "machine_states" in self._sched_params:
            kwargs["machine_states"] = machine_states
        if "t_max" in self._sched_params:
            kwargs["t_max"] = self.t_max
        # Handle both naming conventions for current time
        if "current_step" in self._sched_params:
            kwargs["current_step"] = current_step
        elif "current_time" in self._sched_params:
            kwargs["current_time"] = current_step
        return compute_scheduling_reward(**kwargs)


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
        inventory_total:          float = 0.0,
        delta_ruls:               Optional[List[float]] = None,
        n_auto_cm:                int = 0,
        units_ordered:            float = 0.0,
        inv_below_rop:            bool  = False,
        pre_maint_health:         list = None,
    ) -> Tuple[float, float, float]:
        """Computes r1, r2, R_shared for one timestep."""

        # Criticality weighting for shared penalty
        machine_criticality = compute_machine_criticality(
            newly_failed_machine_ids,
            eligible_map or {},
            n_pending_ops,
        )

        r_shared = self._call_shared(
            newly_failed_machine_ids, machine_criticality,
            n_completions=len(completed_job_ids),
        )
        r1 = self._call_maintenance(
            maintenance_actions, ordering_cost, machine_states,
            r_shared, inventory_total, delta_ruls=delta_ruls,
            n_auto_cm=n_auto_cm,
            units_ordered=units_ordered, inv_below_rop=inv_below_rop,
            pre_maint_health=pre_maint_health,
        )
        r2 = self._call_scheduling(
            jobs, completed_job_ids, assignment,
            machine_states, r_shared, current_step,
            t_max=self.t_max,
        )

        return r1, r2, r_shared