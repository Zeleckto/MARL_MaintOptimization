"""
environments/mfg_env.py
========================
PettingZoo AEC environment — main environment shell.

This file ONLY orchestrates the half-step sequence and calls transition modules.
No physics logic lives here. All physics is in environments/transitions/.

AEC timestep sequence (see architecture doc §3):
    HALF-STEP 1 — Agent 1 (PDM) acts
        - Observe o1_t from current state
        - Apply maintenance + reorder actions
        - Update machine statuses sigma_m
        - State transitions to s_{t+1/2}

    HALF-STEP 2 — Agent 2 (Job Shop) acts
        - Rebuild tripartite graph from s_{t+1/2}
        - Apply scheduling action (j, k, m)
        - Assign operation to machine

    PHYSICS RESOLUTION
        - Weibull degradation tick
        - Processing time decrement
        - Lead time pipeline shift + order arrivals
        - Failure detection + preemption

    REWARD COMPUTATION
        - r1_t, r2_t, R_shared_t all computed here
        - Stored in observations dict for rollout buffer

Usage:
    from environments.mfg_env import ManufacturingEnv
    env = ManufacturingEnv(config)
    observations, infos = env.reset()
    # AEC loop
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        action = policy(obs)
        env.step(action)
"""

import numpy as np
import yaml
import copy
from typing import Dict, List, Optional, Tuple, Any

# PettingZoo AEC base
try:
    from pettingzoo import AECEnv
    from pettingzoo.utils import agent_selector
    PETTINGZOO_AVAILABLE = True
except ImportError:
    # Fallback for testing without pettingzoo
    PETTINGZOO_AVAILABLE = False
    AECEnv = object

from environments.transitions.degradation import (
    MachineState, MachineStatus, DegradationEngine, build_machine_states
)
from environments.transitions.job_dynamics import (
    Job, Operation, OpStatus, JobDynamicsEngine
)
from environments.transitions.resource_dynamics import (
    ResourceState, ResourceManager
)
from environments.transitions.failure_handler import FailureHandler
from environments.spaces.action_spaces import (
    build_agent1_maintenance_mask,
    build_agent1_reorder_mask,
    build_agent2_valid_actions,
    flatten_agent2_actions,
    Q_MAX,
)
from environments.spaces.observation_spaces import (
    compute_agent1_obs_dim,
    OP_FEATURE_DIM, MACHINE_FEATURE_DIM, JOB_FEATURE_DIM,
)
from rewards.reward_fn import RewardFunction


# Agent name constants — PettingZoo uses string ids
AGENT_PDM      = "pdm_agent"
AGENT_JOBSHOP  = "jobshop_agent"
AGENTS         = [AGENT_PDM, AGENT_JOBSHOP]


class ManufacturingEnv(AECEnv if PETTINGZOO_AVAILABLE else object):
    """
    Manufacturing optimization environment following PettingZoo AEC API.

    Two cooperative agents:
        pdm_agent:     Agent 1 — maintenance decisions + resource ordering
        jobshop_agent: Agent 2 — job-machine assignment + sequencing

    Observation spaces:
        pdm_agent:     flat numpy vector (see compute_agent1_obs_dim)
        jobshop_agent: dict with graph components (built by graph_builder.py)

    Action spaces:
        pdm_agent:     dict {'maintenance': [n_mach], 'reorder': [n_con]}
        jobshop_agent: int index into valid (op, machine) pairs, or WAIT
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "manufacturing_v0"}

    def __init__(self, config: dict, render_mode: Optional[str] = None):
        """
        Args:
            config:      Full config dict (loaded from configs/phase1.yaml etc.)
            render_mode: 'human' for pygame window, None for headless
        """
        if PETTINGZOO_AVAILABLE:
            super().__init__()

        self.config      = config
        self.render_mode = render_mode

        # Agent list (PettingZoo requirement)
        self.possible_agents = AGENTS[:]
        self.agents          = AGENTS[:]

        # Episode config
        self.t_max      = config.get("episode", {}).get("t_max_train", 200)
        self.dt         = config.get("episode", {}).get("dt_hours", 8.0)
        self.n_machines = len(config.get("machines", []))
        self.n_jobs     = config.get("jobs", {}).get("n_jobs_train", 20)
        self.stoch_level = config.get("stochasticity_level", 1)

        # Resource requirement matrices [n_machines, n_renewable+n_consumable]
        # These are loaded from config — TBD values for now, set to defaults
        n_ren = len(config["resources"]["renewable"])
        n_con = len(config["resources"]["consumable"])
        n_res = n_ren + n_con

        # Default resource requirements (will be overridden when benchmark instances finalised)
        self.rho_PM = np.ones((self.n_machines, n_res), dtype=float)
        self.rho_CM = np.ones((self.n_machines, n_res), dtype=float) * 2.0
        self.n_renewable = n_ren

        # Precompute max CM consumable need (for reorder masking)
        self.rho_CM_max = self.rho_CM[:, n_ren:].max(axis=0)

        # Initialise transition engines
        self.degradation_engine = DegradationEngine(config)
        self.job_engine         = JobDynamicsEngine(config)
        self.resource_manager   = ResourceManager(config)
        self.failure_handler    = FailureHandler(config)
        self.reward_fn          = RewardFunction(config)

        # Pygame renderer (lazy init)
        self._renderer = None

        # State variables (initialised in reset())
        self.machine_states:  List[MachineState] = []
        self.jobs:            List[Job]           = []
        self.resource_state:  Optional[ResourceState] = None
        self.machine_busy:    List[bool]          = []
        self.current_step:    int                 = 0

        # Per-step tracking
        self._last_maintenance_actions: List[int]               = []
        self._last_ordering_cost:       float                   = 0.0
        self._last_assignment:          Optional[Tuple]         = None
        self._valid_pairs:              List[Tuple[int,int,int]] = []

        # PettingZoo required dicts
        self.rewards      = {a: 0.0  for a in self.possible_agents}
        self.terminations = {a: False for a in self.possible_agents}
        self.truncations  = {a: False for a in self.possible_agents}
        self.infos        = {a: {}    for a in self.possible_agents}
        self._cumulative_rewards = {a: 0.0 for a in self.possible_agents}

        # Agent selector for AEC ordering
        if PETTINGZOO_AVAILABLE:
            self._agent_selector = agent_selector(self.agents)

        # RNG (set properly in reset())
        self._rng = np.random.default_rng(42)

        # Episode stats for logging
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
        """
        Resets environment to initial state for a new episode.

        Returns:
            (observations, infos) dicts keyed by agent name
        """
        # Seed RNG
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Reset episode counter
        self.current_step = 0
        self._episode_failures    = 0
        self._episode_completions = 0

        # Reset agents
        self.agents = self.possible_agents[:]
        self.rewards      = {a: 0.0  for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations  = {a: False for a in self.agents}
        self.infos        = {a: {}    for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}

        # Initialise machine states
        machine_cfgs = self.config.get("machines", [])
        self.machine_states = build_machine_states(machine_cfgs)
        self.machine_busy   = [False] * self.n_machines

        # Initialise resource state
        self.resource_state = self.resource_manager.reset()

        # Generate job batch
        self.jobs = self.job_engine.generate_job_batch(
            n_jobs=self.n_jobs, rng=self._rng
        )

        # Reset tracking
        self._last_maintenance_actions = [0] * self.n_machines
        self._last_ordering_cost       = 0.0
        self._last_assignment          = None

        # Build initial valid pairs for Agent 2
        valid_actions = build_agent2_valid_actions(
            self.jobs, self.machine_states, self.machine_busy
        )
        self._valid_pairs = flatten_agent2_actions(valid_actions)

        # AEC agent selector
        if PETTINGZOO_AVAILABLE:
            self._agent_selector = agent_selector(self.agents)
            self.agent_selection = self._agent_selector.reset()

        # Build initial observations
        observations = {
            AGENT_PDM:     self._build_agent1_obs(),
            AGENT_JOBSHOP: self._build_agent2_obs(),
        }

        return observations, self.infos


    # =========================================================================
    # STEP
    # =========================================================================

    def step(self, action: Any) -> None:
        """
        Processes one agent's action in the AEC sequence.

        PettingZoo AEC: called once per agent per timestep.
        First call = Agent 1 (pdm_agent) half-step.
        Second call = Agent 2 (jobshop_agent) half-step + physics resolution.
        """
        if not self.agents:
            return

        current_agent = self.agent_selection if PETTINGZOO_AVAILABLE else AGENT_PDM

        if current_agent == AGENT_PDM:
            self._step_agent1(action)
        elif current_agent == AGENT_JOBSHOP:
            self._step_agent2(action)
            self._resolve_physics()
            self._compute_rewards()

        # Advance AEC selector
        if PETTINGZOO_AVAILABLE:
            self.agent_selection = self._agent_selector.next()


    def _step_agent1(self, action: dict) -> None:
        """
        HALF-STEP 1: Agent 1 maintenance + reorder actions.
        Updates machine statuses immediately.
        Does NOT compute reward yet.

        Args:
            action: dict with keys:
                'maintenance': np.ndarray [n_machines] each in {0,1,2}
                'reorder':     np.ndarray [n_consumable] each in {0..Q_max}
        """
        if action is None:
            # Random/default action for testing
            maintenance = np.zeros(self.n_machines, dtype=int)
            reorder     = np.zeros(
                len(self.config["resources"]["consumable"]), dtype=float
            )
        else:
            maintenance = np.array(action.get("maintenance",
                np.zeros(self.n_machines, dtype=int)), dtype=int)
            reorder = np.array(action.get("reorder",
                np.zeros(len(self.config["resources"]["consumable"]), dtype=float)))

        # Store for reward computation
        self._last_maintenance_actions = maintenance.tolist()

        # Apply maintenance actions — update machine statuses immediately
        # (degradation.tick will handle the actual state transitions)
        # For now, mark which machines Agent 1 wants to maintain
        # (will be applied in physics resolution)
        self._pending_maintenance = maintenance
        self._pending_reorder     = reorder

        # Rebuild Agent 2's valid actions based on updated machine availability
        # This is the AEC fix — Agent 2 sees post-Agent-1 machine statuses
        valid_actions = build_agent2_valid_actions(
            self.jobs, self.machine_states, self.machine_busy
        )
        self._valid_pairs = flatten_agent2_actions(valid_actions)


    def _step_agent2(self, action: Any) -> None:
        """
        HALF-STEP 2: Agent 2 job-machine assignment.

        Args:
            action: int index into self._valid_pairs
                    len(self._valid_pairs) = WAIT action
        """
        self._last_assignment = None

        if action is None or len(self._valid_pairs) == 0:
            return  # WAIT

        # WAIT action = index beyond valid pairs
        if isinstance(action, (int, np.integer)) and action < len(self._valid_pairs):
            job_id, op_idx, machine_id = self._valid_pairs[int(action)]

            try:
                self.jobs, proc_time = self.job_engine.assign_operation(
                    self.jobs, job_id, op_idx, machine_id, self._rng
                )
                self.machine_busy[machine_id] = True
                self._last_assignment = (job_id, op_idx, machine_id)
            except ValueError as e:
                # Invalid action slipped through masking — log and ignore
                pass


    def _resolve_physics(self) -> None:
        """
        Full physics resolution after both agents have acted.
        Called at end of complete timestep (after Agent 2 half-step).
        """
        # Save old states for failure detection
        old_machine_states = copy.deepcopy(self.machine_states)

        # 1. Weibull degradation + maintenance state transitions
        self.machine_states = self.degradation_engine.tick_all(
            machine_states=self.machine_states,
            operating_flags=self.machine_busy[:],
            rng=self._rng,
            actions_maintenance=self._pending_maintenance.tolist(),
        )

        # 2. Job processing time decrements + completion detection
        self.jobs, completed_ids, freed_machines = self.job_engine.tick(
            jobs=self.jobs,
            current_time=float(self.current_step),
            rng=self._rng,
        )
        self._completed_job_ids = completed_ids
        self._episode_completions += len(completed_ids)

        # Free machines whose operations completed
        for m in freed_machines:
            self.machine_busy[m] = False

        # 3. Failure detection + preemption (shock absorber)
        newly_failed = self.failure_handler.check_failures(
            old_machine_states, self.machine_states
        )
        if newly_failed:
            self.jobs, preempted = self.failure_handler.handle_preemption(
                newly_failed, self.jobs
            )
            # Preempted ops' machines are freed
            self._episode_failures += len(newly_failed)
        self._newly_failed = newly_failed

        # 4. Resource dynamics (inventory update + pipeline shift)
        self.resource_state, self._last_ordering_cost = self.resource_manager.step(
            state=self.resource_state,
            maintenance_actions=self._pending_maintenance.tolist(),
            order_actions=self._pending_reorder,
            rho_PM=self.rho_PM,
            rho_CM=self.rho_CM,
            machines_completing_maint=[],  # TODO: track completing maintenance
            rng=self._rng,
        )

        # 5. Phase 3: sample new job arrivals
        new_jobs = self.job_engine.sample_arrivals(
            current_time=float(self.current_step),
            existing_jobs=self.jobs,
            rng=self._rng,
        )
        self.jobs.extend(new_jobs)

        # Increment timestep
        self.current_step += 1

        # Check termination conditions
        all_done   = all(j.is_complete for j in self.jobs) if self.jobs else False
        timed_out  = self.current_step >= self.t_max

        for agent in self.agents:
            self.terminations[agent] = all_done
            self.truncations[agent]  = timed_out and not all_done


    def _compute_rewards(self) -> None:
        """Compute and store rewards for both agents."""
        # Build eligible_map for criticality weighting: machine -> ops that need it
        eligible_map = {}
        n_pending = 0
        for job in self.jobs:
            for op in job.operations:
                if op.status in (0, 1, 2):   # PENDING, READY, IN_PROGRESS
                    n_pending += 1
                    for m in op.eligible_machines:
                        eligible_map.setdefault(m, []).append((job.job_id, op.op_idx))

        r1, r2, r_shared = self.reward_fn.compute(
            maintenance_actions      = self._last_maintenance_actions,
            ordering_cost            = self._last_ordering_cost,
            machine_states           = self.machine_states,
            newly_failed_machine_ids = self._newly_failed,
            jobs                     = self.jobs,
            completed_job_ids        = self._completed_job_ids,
            assignment               = self._last_assignment,
            current_step             = self.current_step,
            eligible_map             = eligible_map,
            n_pending_ops            = n_pending,
        )
        self.rewards[AGENT_PDM]     = r1
        self.rewards[AGENT_JOBSHOP] = r2
        self._last_r_shared         = r_shared

        # Accumulate for episode return tracking
        self._cumulative_rewards[AGENT_PDM]     += r1
        self._cumulative_rewards[AGENT_JOBSHOP] += r2


    # =========================================================================
    # OBSERVATIONS
    # =========================================================================

    def _build_agent1_obs(self) -> np.ndarray:
        """
        Builds Agent 1's flat observation vector.

        Structure:
            [machine_features (15 * n_machines),
             resource_state (flat vector),
             job_summary (5 stats)]
        """
        # Per-machine features
        machine_feats = np.concatenate([
            s.to_feature_vector() for s in self.machine_states
        ])

        # Resource state
        resource_feats = self.resource_state.to_flat_vector()

        # Job summary (5 aggregate stats)
        active_jobs  = self.job_engine.get_active_jobs(self.jobs)
        n_active     = len(active_jobs)
        n_at_risk    = sum(
            1 for j in active_jobs
            if j.due_date - self.current_step < j.n_ops * 2
        )
        avg_comp     = (
            np.mean([j.completion_ratio for j in active_jobs])
            if active_jobs else 0.0
        )
        avg_slack    = (
            np.mean([j.due_date - self.current_step for j in active_jobs])
            / self.t_max if active_jobs else 0.0
        )
        n_ready_ops  = len(self.job_engine.get_ready_ops(self.jobs))

        job_summary = np.array([
            n_active     / max(self.n_jobs, 1),
            n_at_risk    / max(self.n_jobs, 1),
            float(avg_comp),
            float(np.clip(avg_slack, -1, 1)),
            n_ready_ops  / max(self.n_jobs * 3, 1),
        ], dtype=np.float32)

        return np.concatenate([machine_feats, resource_feats, job_summary])


    def _build_agent2_obs(self) -> dict:
        """
        Builds Agent 2's graph observation as a dict of numpy arrays.
        graph_builder.py converts this to PyG HeteroData.

        Returns dict with keys:
            'op_features':         [n_ops, 10]
            'machine_features':    [n_machines, 15]
            'job_features':        [n_jobs, 7]
            'edge_op_mach':        [2, n_edges_om] (src, dst indices)
            'edge_attr_op_mach':   [n_edges_om, 2]
            'edge_mach_job':       [2, n_edges_mj]
            'edge_attr_mach_job':  [n_edges_mj, 2]
            'edge_op_job':         [2, n_edges_oj]
            'edge_attr_op_job':    [n_edges_oj, 2]
            'valid_pairs':         list of (job_id, op_idx, machine_id)
        """
        active_jobs = self.job_engine.get_active_jobs(self.jobs)
        t           = float(self.current_step)

        # --- Op nodes ---
        # Collect all non-DONE operations
        pending_ops  = []
        op_to_idx    = {}   # (job_id, op_idx) -> node index
        n_total_ops  = max(sum(j.n_ops for j in active_jobs), 1)

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

        # --- Machine nodes ---
        machine_features = np.stack([
            s.to_feature_vector() for s in self.machine_states
        ])

        # --- Job nodes ---
        if not active_jobs:
            job_features = np.zeros((1, JOB_FEATURE_DIM), dtype=np.float32)
        else:
            job_features = np.stack([j.to_feature_vector(t) for j in active_jobs])

        job_to_idx = {j.job_id: i for i, j in enumerate(active_jobs)}

        # --- Edges: Op -> Machine (eligible and available) ---
        edge_om_src, edge_om_dst, edge_attr_om = [], [], []
        for (job_id, op_idx), op_node_idx in op_to_idx.items():
            op, _ = pending_ops[op_node_idx]
            for m in op.eligible_machines:
                if self.machine_states[m].status == MachineStatus.OP:
                    edge_om_src.append(op_node_idx)
                    edge_om_dst.append(m)
                    # Edge features: normalised proc time, compatibility=1.0
                    proc_t = op.nominal_proc_times.get(m, 8.0) / 50.0
                    edge_attr_om.append([min(proc_t, 1.0), 1.0])

        # --- Edges: Machine -> Job (machine serving a job's op) ---
        edge_mj_src, edge_mj_dst, edge_attr_mj = [], [], []
        for job in active_jobs:
            j_idx = job_to_idx[job.job_id]
            for op in job.operations:
                if op.status == OpStatus.IN_PROGRESS and op.assigned_machine is not None:
                    m_idx = op.assigned_machine
                    progress = 1.0 - (op.remaining_time / max(op.actual_proc_time, 0.01))
                    edge_mj_src.append(m_idx)
                    edge_mj_dst.append(j_idx)
                    edge_attr_mj.append([min(max(progress, 0.0), 1.0),
                                         op.remaining_time / self.t_max])

        # --- Edges: Op -> Job (structural, always exists) ---
        edge_oj_src, edge_oj_dst, edge_attr_oj = [], [], []
        for (job_id, op_idx), op_node_idx in op_to_idx.items():
            if job_id in job_to_idx:
                j_idx = job_to_idx[job_id]
                op, _ = pending_ops[op_node_idx]
                n_ops  = next(j.n_ops for j in active_jobs if j.job_id == job_id)
                edge_oj_src.append(op_node_idx)
                edge_oj_dst.append(j_idx)
                edge_attr_oj.append([
                    op.op_idx / max(n_ops - 1, 1),
                    float(op.status == OpStatus.READY),
                ])

        def safe_edge(src, dst, attr, feat_dim):
            if not src:
                return (np.zeros((2, 0), dtype=np.int64),
                        np.zeros((0, feat_dim), dtype=np.float32))
            return (np.array([src, dst], dtype=np.int64),
                    np.array(attr, dtype=np.float32))

        e_om, a_om = safe_edge(edge_om_src, edge_om_dst, edge_attr_om, 2)
        e_mj, a_mj = safe_edge(edge_mj_src, edge_mj_dst, edge_attr_mj, 2)
        e_oj, a_oj = safe_edge(edge_oj_src, edge_oj_dst, edge_attr_oj, 2)

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
    # PETTINGZOO REQUIRED METHODS
    # =========================================================================

    def observe(self, agent: str) -> Any:
        """Returns current observation for given agent."""
        if agent == AGENT_PDM:
            return self._build_agent1_obs()
        else:
            return self._build_agent2_obs()


    def last(self, observe: bool = True):
        """Returns (obs, reward, termination, truncation, info) for current agent."""
        agent = self.agent_selection if PETTINGZOO_AVAILABLE else AGENT_PDM
        obs   = self.observe(agent) if observe else None
        return (
            obs,
            self.rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )


    def render(self) -> Optional[np.ndarray]:
        """Renders environment using pygame. Lazy-initialised."""
        if self.render_mode is None:
            return None

        if self._renderer is None:
            try:
                from environments.rendering.pygame_renderer import PygameRenderer
                self._renderer = PygameRenderer(self.config)
            except ImportError:
                print("Pygame not available — rendering disabled")
                return None

        return self._renderer.render(
            machine_states=self.machine_states,
            jobs=self.jobs,
            resource_state=self.resource_state,
            current_step=self.current_step,
            valid_pairs=self._valid_pairs,
        )


    def close(self) -> None:
        """Cleanup."""
        if self._renderer is not None:
            self._renderer.close()


    def action_space(self, agent: str):
        """Returns action space for given agent (informational)."""
        return None  # Dynamic spaces not supported by gym Space API


    def observation_space(self, agent: str):
        """Returns observation space for given agent (informational)."""
        return None