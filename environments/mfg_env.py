"""
environments/mfg_env.py
========================
PettingZoo AEC environment — updated for Phase 0+1+2 redesign.

Changes:
  - snapshot prev_ruls BEFORE physics tick (needed for ΔRUL reward)
  - Phase 3 arrival hook gated behind stochasticity_level >= 3
  - Agent 1 obs uses new resource pipeline format
  - H is now from Weibull (auto via updated degradation.py)
  - No h_PM_threshold anywhere in this file

AEC timestep sequence:
    _step_agent1(action1)      Agent 1 maintenance + reorder
    _step_agent2(action2_idx)  Agent 2 job assignment
    _resolve_physics()         Weibull tick, job updates, failures
    _compute_rewards()         r1, r2 computed (uses prev_ruls)

Call these 4 methods directly from mappo_trainer. Do NOT use env.step()
from outside — PettingZoo's AEC selector causes ordering issues.
"""

import numpy as np
import copy
from typing import Dict, List, Optional, Tuple, Any

try:
    from pettingzoo import AECEnv
    from pettingzoo.utils.agent_selector import agent_selector
    PETTINGZOO_AVAILABLE = True
except ImportError:
    PETTINGZOO_AVAILABLE = False
    AECEnv = object

from environments.transitions.degradation import (
    MachineState, MachineStatus, DegradationEngine, build_machine_states, estimate_rul
)
from environments.transitions.job_dynamics import (
    Job, Operation, OpStatus, JobDynamicsEngine
)
from environments.transitions.resource_dynamics import (
    ResourceState, ResourceManager
)
from environments.transitions.failure_handler import FailureHandler
from environments.spaces.action_spaces import (
    build_agent2_valid_actions, flatten_agent2_actions,
)
from environments.spaces.observation_spaces import (
    MACHINE_FEATURE_DIM, OP_FEATURE_DIM, JOB_FEATURE_DIM,
    JOB_SUMMARY_DIM, compute_agent1_obs_dim,
)
from rewards.reward_fn import RewardFunction

AGENT_PDM     = "pdm_agent"
AGENT_JOBSHOP = "jobshop_agent"
AGENTS        = [AGENT_PDM, AGENT_JOBSHOP]


class ManufacturingEnv(AECEnv if PETTINGZOO_AVAILABLE else object):
    """
    Manufacturing optimization environment — PettingZoo AEC API.
    Two cooperative agents: PDM (Agent 1) and Job Shop (Agent 2).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "manufacturing_v0"}

    def __init__(self, config: dict, render_mode: Optional[str] = None):
        if PETTINGZOO_AVAILABLE:
            super().__init__()

        self.config      = config
        self.render_mode = render_mode

        self.possible_agents = AGENTS[:]
        self.agents          = AGENTS[:]

        self.t_max       = config.get("episode", {}).get("t_max_train", 150)
        self.dt          = config.get("episode", {}).get("dt_hours", 8.0)
        self.n_machines  = len(config.get("machines", []))
        self.n_jobs      = config.get("jobs", {}).get("n_jobs_train", 40)
        self.stoch_level = config.get("stochasticity_level", 1)

        # Resource dimensions
        res_cfg          = config.get("resources", {})
        self._n_renewable  = len(res_cfg.get("renewable", []))
        self._n_consumable = len(res_cfg.get("consumable", []))
        self._max_lead = max(
            (r.get("lead_time_shifts", 5) for r in res_cfg.get("consumable", [])),
            default=7
        )

        # Resource requirement vectors (from config)
        req = config.get("resource_requirements", {})
        n_res = self._n_renewable + self._n_consumable
        self.rho_PM       = np.ones((self.n_machines, n_res), dtype=float)
        self.rho_CM       = np.ones((self.n_machines, n_res), dtype=float) * 2.0
        self.n_renewable  = self._n_renewable

        # Transition engines
        self.degradation_engine = DegradationEngine(config)
        self.job_engine         = JobDynamicsEngine(config)
        self.resource_manager   = ResourceManager(config)
        self.failure_handler    = FailureHandler(config)
        self.reward_fn          = RewardFunction(config)

        # State variables (set in reset())
        self.machine_states:  List[MachineState] = []
        self.jobs:            List[Job]           = []
        self.resource_state:  Optional[ResourceState] = None
        self.machine_busy:    List[bool]          = []
        self.current_step:    int                 = 0

        # Per-step tracking
        self._last_maintenance_actions: List[int]   = []
        self._last_ordering_cost:       float       = 0.0
        self._last_assignment:          Optional[Tuple] = None
        self._valid_pairs:              list        = []
        self._pending_maintenance:      np.ndarray  = np.zeros(self.n_machines, dtype=int)
        self._pending_reorder:          np.ndarray  = np.zeros(self._n_consumable)
        self._prev_ruls:                List[float] = []   # ΔRUL snapshot
        self._newly_failed:             List[int]   = []
        self._completed_job_ids:        List[int]   = []

        # PettingZoo required
        self.rewards      = {a: 0.0  for a in AGENTS}
        self.terminations = {a: False for a in AGENTS}
        self.truncations  = {a: False for a in AGENTS}
        self.infos        = {a: {}   for a in AGENTS}
        self._cumulative_rewards = {a: 0.0 for a in AGENTS}

        if PETTINGZOO_AVAILABLE:
            self._agent_selector = agent_selector(self.agents)

        self._rng = np.random.default_rng(42)
        self._episode_failures    = 0
        self._episode_completions = 0

    # =========================================================================
    # RESET
    # =========================================================================
    def reset(
        self,
        seed:    Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict, Dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.current_step         = 0
        self._episode_failures    = 0
        self._episode_completions = 0

        self.agents = self.possible_agents[:]
        self.rewards      = {a: 0.0  for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations  = {a: False for a in self.agents}
        self.infos        = {a: {}   for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}

        machine_cfgs = self.config.get("machines", [])
        self.machine_states = build_machine_states(machine_cfgs)
        self.machine_busy   = [False] * self.n_machines

        self.resource_state = self.resource_manager.reset()

        self.jobs = self.job_engine.generate_job_batch(
            n_jobs=self.n_jobs, rng=self._rng
        )

        self._last_maintenance_actions = [0] * self.n_machines
        self._last_ordering_cost       = 0.0
        self._last_assignment          = None
        self._newly_failed             = []
        self._completed_job_ids        = []
        self._prev_ruls = [estimate_rul(s) for s in self.machine_states]

        valid_actions = build_agent2_valid_actions(
            self.jobs, self.machine_states, self.machine_busy
        )
        self._valid_pairs = flatten_agent2_actions(valid_actions)

        if PETTINGZOO_AVAILABLE:
            self._agent_selector = agent_selector(self.agents)
            self.agent_selection = self._agent_selector.reset()

        observations = {
            AGENT_PDM:     self._build_agent1_obs(),
            AGENT_JOBSHOP: self._build_agent2_obs(),
        }
        return observations, self.infos

    # =========================================================================
    # HALF-STEPS (call these directly from trainer, not env.step())
    # =========================================================================
    def _step_agent1(self, action: dict) -> None:
        """
        Agent 1 half-step: receives maintenance + reorder actions.
        Stores them for physics resolution. Does NOT compute reward yet.
        """
        if action is None:
            maintenance = np.zeros(self.n_machines, dtype=int)
            reorder     = np.zeros(self._n_consumable, dtype=float)
        else:
            maintenance = np.array(
                action.get("maintenance", np.zeros(self.n_machines, dtype=int)),
                dtype=int
            )
            reorder = np.array(
                action.get("reorder", np.zeros(self._n_consumable, dtype=float))
            )

        self._last_maintenance_actions = maintenance.tolist()
        self._pending_maintenance      = maintenance
        self._pending_reorder          = reorder

        # IMPORTANT: snapshot RUL BEFORE physics tick
        self._prev_ruls = [estimate_rul(s) for s in self.machine_states]

        # Rebuild valid pairs for Agent 2 (reflects Agent 1's intent)
        valid_actions = build_agent2_valid_actions(
            self.jobs, self.machine_states, self.machine_busy
        )
        self._valid_pairs = flatten_agent2_actions(valid_actions)

    def _step_agent2(self, action: Any) -> None:
        """Agent 2 half-step: job-machine assignment or WAIT."""
        self._last_assignment = None

        if action is None or len(self._valid_pairs) == 0:
            return

        if isinstance(action, (int, np.integer)) and int(action) < len(self._valid_pairs):
            job_id, op_idx, machine_id = self._valid_pairs[int(action)]
            try:
                self.jobs, proc_time = self.job_engine.assign_operation(
                    self.jobs, job_id, op_idx, machine_id, self._rng
                )
                self.machine_busy[machine_id] = True
                self._last_assignment = (job_id, op_idx, machine_id)
            except ValueError:
                pass  # invalid action slipped masking — ignore

    def _resolve_physics(self) -> None:
        """
        Full physics: degradation, job ticks, failures, resources, arrivals.
        """
        old_states = copy.deepcopy(self.machine_states)

        # 1. Weibull degradation + maintenance
        self.machine_states = self.degradation_engine.tick_all(
            machine_states       = self.machine_states,
            operating_flags      = self.machine_busy[:],
            rng                  = self._rng,
            actions_maintenance  = self._pending_maintenance.tolist(),
        )

        # 2. Job processing tick
        self.jobs, completed_ids, freed_machines = self.job_engine.tick(
            jobs         = self.jobs,
            current_time = float(self.current_step),
            rng          = self._rng,
        )
        self._completed_job_ids = completed_ids
        self._episode_completions += len(completed_ids)
        for m in freed_machines:
            self.machine_busy[m] = False

        # 3. Failure detection + shock absorber preemption
        newly_failed = self.failure_handler.check_failures(old_states, self.machine_states)
        if newly_failed:
            self.jobs, _ = self.failure_handler.handle_preemption(newly_failed, self.jobs)
            self._episode_failures += len(newly_failed)
        self._newly_failed = newly_failed

        # 4. Resource dynamics
        self.resource_state, self._last_ordering_cost = self.resource_manager.step(
            state               = self.resource_state,
            maintenance_actions = self._pending_maintenance.tolist(),
            order_actions       = self._pending_reorder,
            rho_PM              = self.rho_PM,
            rho_CM              = self.rho_CM,
            machines_completing_maint = [],
            rng                 = self._rng,
        )

        # 5. Phase 3: Poisson job arrivals
        new_jobs = self.job_engine.sample_arrivals(
            current_time  = float(self.current_step),
            existing_jobs = self.jobs,
            rng           = self._rng,
        )
        self.jobs.extend(new_jobs)

        self.current_step += 1

        all_done = all(j.is_complete for j in self.jobs) if self.jobs else False
        timed_out = self.current_step >= self.t_max

        for agent in self.agents:
            self.terminations[agent] = all_done
            self.truncations[agent]  = timed_out and not all_done

    def _compute_rewards(self) -> None:
        """Compute and store rewards. Uses _prev_ruls for ΔRUL signal."""
        # Get current consumable inventory for holding cost
        consumable_inv = None
        if self.resource_state is not None:
            consumable_inv = list(self.resource_state.consumable_inventory)

        r1, r2, r_shared = self.reward_fn.compute(
            maintenance_actions      = self._last_maintenance_actions,
            ordering_cost            = self._last_ordering_cost,
            machine_states           = self.machine_states,
            newly_failed_machine_ids = self._newly_failed,
            jobs                     = self.jobs,
            completed_job_ids        = self._completed_job_ids,
            assignment               = self._last_assignment,
            current_time             = float(self.current_step),
            prev_ruls                = self._prev_ruls,
            consumable_inventory     = consumable_inv,
        )
        self.rewards[AGENT_PDM]     = r1
        self.rewards[AGENT_JOBSHOP] = r2
        self._last_r_shared         = r_shared
        self._cumulative_rewards[AGENT_PDM]     += r1
        self._cumulative_rewards[AGENT_JOBSHOP] += r2

    # =========================================================================
    # OBSERVATIONS
    # =========================================================================
    def _build_agent1_obs(self) -> np.ndarray:
        """
        Agent 1 flat observation:
        [machine_features | resource_renewable | resource_consumable
         | resource_pipeline | job_summary]
        """
        machine_feats = np.concatenate([
            s.to_feature_vector() for s in self.machine_states
        ])

        # Resource features
        rs = self.resource_state
        if rs is not None:
            # Renewable: [available/cap, cap/max_cap] per resource
            ren_feats = np.concatenate([
                np.array([
                    rs.renewable_available[i] / max(rs.renewable_capacity[i], 1),
                    rs.renewable_available[i] / max(rs.renewable_capacity[i], 1),
                ], dtype=np.float32)
                for i in range(self._n_renewable)
            ]) if self._n_renewable > 0 else np.zeros(0, dtype=np.float32)

            # Consumable: [inventory/cap, pipeline_sum/cap] per resource
            con_feats_list = []
            for i, r_cfg in enumerate(self.config.get("resources", {}).get("consumable", [])):
                cap = float(r_cfg.get("initial_inventory", 10)) * 2.0
                inv = rs.consumable_inventory[i] / max(cap, 1)
                pipeline = rs.pending_orders[i].sum() / max(cap, 1)
                con_feats_list.extend([inv, pipeline])
            con_feats = np.array(con_feats_list, dtype=np.float32)

            # Pipeline: full [n_consumable, max_lead] flattened
            pipeline_feats = (rs.pending_orders / 10.0).astype(np.float32).flatten()
            pipeline_feats = np.clip(pipeline_feats, 0, 2.0)
        else:
            ren_feats      = np.zeros(self._n_renewable * 2, dtype=np.float32)
            con_feats      = np.zeros(self._n_consumable * 2, dtype=np.float32)
            pipeline_feats = np.zeros(self._n_consumable * self._max_lead, dtype=np.float32)

        # Job summary (6 stats)
        active_jobs = self.job_engine.get_active_jobs(self.jobs)
        n_active    = len(active_jobs)
        n_at_risk   = sum(
            1 for j in active_jobs
            if (j.due_date - self.current_step) < sum(
                min(op.nominal_proc_times.values()) if op.nominal_proc_times else 2.0
                for op in j.operations
                if op.status not in (3,)
            )
        )
        avg_comp  = float(np.mean([j.completion_ratio for j in active_jobs])) \
                    if active_jobs else 0.0
        avg_slack = float(np.mean([j.due_date - self.current_step for j in active_jobs])) \
                    / self.t_max if active_jobs else 0.0
        n_ready   = len(self.job_engine.get_ready_ops(self.jobs))
        avg_health = float(np.mean([s.health / 100.0 for s in self.machine_states]))

        job_summary = np.array([
            n_active  / max(self.n_jobs, 1),
            n_at_risk / max(self.n_jobs, 1),
            avg_comp,
            np.clip(avg_slack, -1, 1),
            n_ready   / max(self.n_jobs * 5, 1),
            avg_health,
        ], dtype=np.float32)

        return np.concatenate([machine_feats, ren_feats, con_feats, pipeline_feats, job_summary])

    def _build_agent2_obs(self) -> dict:
        """
        Agent 2 graph observation as dict of numpy arrays.
        graph_builder.py converts this to PyG HeteroData.
        """
        active_jobs = self.job_engine.get_active_jobs(self.jobs)
        t           = float(self.current_step)

        # Op nodes
        pending_ops = []
        op_to_idx   = {}
        n_total_ops = max(sum(j.n_ops for j in active_jobs), 1)

        for job in active_jobs:
            for op in job.operations:
                if op.status != OpStatus.DONE:
                    idx = len(pending_ops)
                    op_to_idx[(job.job_id, op.op_idx)] = idx
                    pending_ops.append((op, job.due_date))

        if not pending_ops:
            op_features = np.zeros((1, OP_FEATURE_DIM), dtype=np.float32)
        else:
            op_features = np.stack([
                op.to_feature_vector(due, t, self.n_machines, n_total_ops)
                for op, due in pending_ops
            ])

        # Machine nodes
        machine_features = np.stack([s.to_feature_vector() for s in self.machine_states])

        # Job nodes
        if not active_jobs:
            job_features = np.zeros((1, JOB_FEATURE_DIM), dtype=np.float32)
        else:
            job_features = np.stack([j.to_feature_vector(t) for j in active_jobs])

        job_to_idx = {j.job_id: i for i, j in enumerate(active_jobs)}

        # Edges: Op -> Machine
        edge_om_src, edge_om_dst, edge_attr_om = [], [], []
        for (job_id, op_idx), op_node_idx in op_to_idx.items():
            op, _ = pending_ops[op_node_idx]
            for m in op.eligible_machines:
                if self.machine_states[m].status == MachineStatus.OP:
                    proc_t = op.nominal_proc_times.get(m, 4.0) / 8.0
                    edge_om_src.append(op_node_idx)
                    edge_om_dst.append(m)
                    edge_attr_om.append([min(proc_t, 2.0), 1.0])

        # Edges: Machine -> Job
        edge_mj_src, edge_mj_dst, edge_attr_mj = [], [], []
        for job in active_jobs:
            j_idx = job_to_idx[job.job_id]
            for op in job.operations:
                if op.status == OpStatus.IN_PROGRESS and op.assigned_machine is not None:
                    m_idx    = op.assigned_machine
                    progress = 1.0 - (op.remaining_time / max(op.actual_proc_time, 0.01))
                    edge_mj_src.append(m_idx)
                    edge_mj_dst.append(j_idx)
                    edge_attr_mj.append([
                        np.clip(progress, 0.0, 1.0),
                        op.remaining_time / self.t_max,
                    ])

        # Edges: Op -> Job (structural)
        edge_oj_src, edge_oj_dst, edge_attr_oj = [], [], []
        for (job_id, op_idx), op_node_idx in op_to_idx.items():
            if job_id in job_to_idx:
                j_idx = job_to_idx[job_id]
                op, _ = pending_ops[op_node_idx]
                n_ops = next(j.n_ops for j in active_jobs if j.job_id == job_id)
                edge_oj_src.append(op_node_idx)
                edge_oj_dst.append(j_idx)
                edge_attr_oj.append([
                    op.op_idx / max(n_ops - 1, 1),
                    float(op.status == OpStatus.READY),
                ])

        def _safe_edge(src, dst, attr, feat_dim):
            if not src:
                return (np.zeros((2, 0), dtype=np.int64),
                        np.zeros((0, feat_dim), dtype=np.float32))
            return (np.array([src, dst], dtype=np.int64),
                    np.array(attr, dtype=np.float32))

        e_om, a_om = _safe_edge(edge_om_src, edge_om_dst, edge_attr_om, 2)
        e_mj, a_mj = _safe_edge(edge_mj_src, edge_mj_dst, edge_attr_mj, 2)
        e_oj, a_oj = _safe_edge(edge_oj_src, edge_oj_dst, edge_attr_oj, 2)

        return {
            "op_features":        op_features,
            "machine_features":   machine_features,
            "job_features":       job_features,
            "edge_op_mach":       e_om,
            "edge_attr_op_mach":  a_om,
            "edge_mach_job":      e_mj,
            "edge_attr_mach_job": a_mj,
            "edge_op_job":        e_oj,
            "edge_attr_op_job":   a_oj,
            "valid_pairs":        self._valid_pairs,
        }

    # =========================================================================
    # PETTINGZOO API
    # =========================================================================
    def step(self, action: Any) -> None:
        """PettingZoo compatibility. Use direct methods in trainer."""
        if not self.agents:
            return
        current_agent = self.agent_selection if PETTINGZOO_AVAILABLE else AGENT_PDM
        if current_agent == AGENT_PDM:
            self._step_agent1(action)
        elif current_agent == AGENT_JOBSHOP:
            self._step_agent2(action)
            self._resolve_physics()
            self._compute_rewards()
        if PETTINGZOO_AVAILABLE:
            self.agent_selection = self._agent_selector.next()

    def observe(self, agent: str) -> Any:
        if agent == AGENT_PDM:
            return self._build_agent1_obs()
        return self._build_agent2_obs()

    def last(self, observe: bool = True):
        agent = self.agent_selection if PETTINGZOO_AVAILABLE else AGENT_PDM
        obs   = self.observe(agent) if observe else None
        return (obs, self.rewards[agent], self.terminations[agent],
                self.truncations[agent], self.infos[agent])

    def render(self):
        return None

    def close(self):
        pass

    def action_space(self, agent):
        return None

    def observation_space(self, agent):
        return None
