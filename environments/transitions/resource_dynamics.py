"""
environments/transitions/resource_dynamics.py
==============================================
Inventory management, replenishment pipeline, and resource allocation tracking.

THE MARKOV FIX (from architecture doc):
    Constraint C32: I_{r,t+1} = I_{r,t} - consumed_{r,t} + Q_{r, t-L_r}
    The replenishment Q_{r, t-L_r} depends on an order placed L_r steps ago.
    Without tracking the FULL pipeline, the state is not Markov-sufficient.

    Solution: pending_orders[r, lag] stores units ordered 'lag' steps ago.
    This shifts left every timestep. When lag == L_r, order arrives in inventory.

Two resource categories (Table 3.5 in report):
    Renewable:  Technicians, Tools, Maintenance Bays.
                Capacity K_r. Freed after each maintenance event. No inventory.
    Consumable: Spare Parts, Lubricants, Consumable Tools.
                Inventory I_r,t. Depleted by maintenance. Replenished after lead time.

Agent 1 action for resources:
    a_order_r in {0, 1, ..., Q_max} per consumable resource.
    Q_max = 2 * max(rho_CM_{m,r}) -- enough for two corrective maintenances.

Usage:
    from environments.transitions.resource_dynamics import ResourceState, ResourceManager
    manager = ResourceManager(config)
    state = manager.reset()
    state = manager.step(state, maintenance_actions, order_actions, rng)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np

from utils.distributions import sample_lead_time, sample_resource_requirement


# =============================================================================
# CONSTANTS — resource types
# =============================================================================

class ResourceType:
    RENEWABLE  = 0
    CONSUMABLE = 1


# =============================================================================
# RESOURCE STATE DATACLASS
# =============================================================================

@dataclass
class ResourceState:
    """
    Complete resource state at one timestep.
    Goes into Agent 1's observation and the centralized critic.

    Attributes:
        renewable_available:  [n_renewable] — free capacity right now
        renewable_capacity:   [n_renewable] — total fixed capacity (for normalisation)
        consumable_inventory: [n_consumable] — current stock
        pending_orders:       [n_consumable, max_lead_time]
                              pending_orders[r, 0]   = arrives NEXT step
                              pending_orders[r, k]   = arrives in k+1 steps
                              pending_orders[r, L-1] = just ordered this step
        lead_times:           [n_consumable] — L_r per resource (fixed from config)
        max_lead_time:        int — width of pending_orders array
    """
    renewable_available:  np.ndarray   # [n_renewable], int
    renewable_capacity:   np.ndarray   # [n_renewable], int
    consumable_inventory: np.ndarray   # [n_consumable], float
    pending_orders:       np.ndarray   # [n_consumable, max_lead_time], float
    lead_times:           np.ndarray   # [n_consumable], int
    max_lead_time:        int

    def to_flat_vector(self) -> np.ndarray:
        """
        Flattens all resource state to 1D float32 array for Agent 1 observation.

        Layout:
            [renewable_available / capacity,        # [n_renewable] normalised to [0,1]
             consumable_inventory / 100.0,           # [n_consumable] normalised
             pending_orders.flatten() / 100.0]       # [n_consumable * max_lead_time]

        Total dim = n_renewable + n_consumable * (1 + max_lead_time)
        For our config: 3 + 3*(1+5) = 3 + 18 = 21 dims
        """
        renewable_norm  = self.renewable_available / np.maximum(
            self.renewable_capacity, 1
        )
        consumable_norm = self.consumable_inventory / 100.0
        orders_norm     = self.pending_orders.flatten() / 100.0

        return np.concatenate([
            renewable_norm, consumable_norm, orders_norm
        ]).astype(np.float32)

    @property
    def obs_dim(self) -> int:
        """Flat observation vector dimension — use this in observation_spaces.py."""
        n_r = len(self.renewable_available)
        n_c = len(self.consumable_inventory)
        return n_r + n_c * (1 + self.max_lead_time)

    def can_do_maintenance(
        self,
        rho_renewable: np.ndarray,   # [n_renewable] units needed
        rho_consumable: np.ndarray,  # [n_consumable] units needed
    ) -> bool:
        """
        Checks whether sufficient resources exist for a maintenance event.
        Used by action masking in action_spaces.py.

        Args:
            rho_renewable:  resource units needed from each renewable type
            rho_consumable: resource units needed from each consumable type

        Returns:
            True if all requirements can be met, False otherwise
        """
        renewable_ok  = np.all(self.renewable_available  >= rho_renewable)
        consumable_ok = np.all(self.consumable_inventory >= rho_consumable)
        return bool(renewable_ok and consumable_ok)

    def projected_consumable_need(self, horizon: int, rho_CM: np.ndarray) -> np.ndarray:
        """
        Estimates consumable need over next 'horizon' steps assuming one CM per step.
        Used by reorder action masking to prevent over-ordering.

        Args:
            horizon:  number of steps to look ahead (use max_lead_time)
            rho_CM:   [n_consumable] units per CM event

        Returns:
            [n_consumable] projected need over horizon
        """
        projected_need = rho_CM * horizon
        # subtract what's already in pipeline
        pipeline_supply = self.pending_orders.sum(axis=1)
        net_need = projected_need - self.consumable_inventory - pipeline_supply
        return np.maximum(net_need, 0.0)


# =============================================================================
# RESOURCE MANAGER
# =============================================================================

class ResourceManager:
    """
    Manages all resource dynamics for the environment.

    Called once per full timestep (after both agents have acted,
    during physics resolution in mfg_env.py).

    Sequence within step():
        1. Consume resources for maintenance actions taken this step
        2. Shift pending_orders pipeline left by 1 (orders age by 1 step)
        3. Deliver orders that have arrived (lag=0 after shift)
        4. Inject new orders from Agent 1's order actions
        5. Free renewable resources from completed maintenance
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Full config dict. Reads:
                resources.renewable:  list of {resource_id, name, capacity}
                resources.consumable: list of {resource_id, name,
                                               initial_inventory, lead_time_shifts,
                                               reorder_cost}
                stochasticity_level: int (affects lead time sampling in Phase 3)
        """
        self.stoch_level = config.get("stochasticity_level", 1)

        # --- Parse renewable resources ---
        ren_cfgs = config["resources"]["renewable"]
        self.n_renewable     = len(ren_cfgs)
        self.renewable_names = [r["name"] for r in ren_cfgs]
        self.renewable_caps  = np.array([r["capacity"] for r in ren_cfgs], dtype=int)

        # --- Parse consumable resources ---
        con_cfgs = config["resources"]["consumable"]
        self.n_consumable       = len(con_cfgs)
        self.consumable_names   = [r["name"] for r in con_cfgs]
        self.initial_inventories = np.array(
            [r["initial_inventory"] for r in con_cfgs], dtype=float
        )
        self.nominal_lead_times = np.array(
            [r["lead_time_shifts"] for r in con_cfgs], dtype=int
        )
        self.reorder_costs = np.array(
            [r["reorder_cost"] for r in con_cfgs], dtype=float
        )

        # max_lead_time determines width of pending_orders array
        # must be fixed so observation dimension is constant
        self.max_lead_time = int(np.max(self.nominal_lead_times))

        # Q_max: maximum units orderable per step per resource
        # set conservatively — will be refined when rho_CM matrix is finalised
        self.Q_max = 10

        # sigma_log for lead time noise (Phase 3)
        self.sigma_log_lead = config.get("processing", {}).get("sigma_log", 0.15)


    def reset(self) -> ResourceState:
        """
        Returns initial ResourceState at episode start.
        All renewables at full capacity, consumables at initial inventory,
        pending_orders all zeros (no orders in flight).
        """
        return ResourceState(
            renewable_available  = self.renewable_caps.copy(),
            renewable_capacity   = self.renewable_caps.copy(),
            consumable_inventory = self.initial_inventories.copy(),
            pending_orders       = np.zeros(
                (self.n_consumable, self.max_lead_time), dtype=float
            ),
            lead_times  = self.nominal_lead_times.copy(),
            max_lead_time = self.max_lead_time,
        )


    def step(
        self,
        state:               ResourceState,
        maintenance_actions: List[int],      # [n_machines] 0=none,1=PM,2=CM
        order_actions:       np.ndarray,     # [n_consumable] units to order
        rho_PM:              np.ndarray,     # [n_machines, n_renewable+n_consumable]
        rho_CM:              np.ndarray,     # [n_machines, n_renewable+n_consumable]
        machines_completing_maint: List[int], # machine ids finishing maint this step
        rng:                 np.random.Generator,
    ) -> Tuple[ResourceState, float]:
        """
        Advances resource state by one full timestep.

        Args:
            state:                     Current ResourceState
            maintenance_actions:       Agent 1's action per machine [n_machines]
            order_actions:             Agent 1's reorder quantities [n_consumable]
            rho_PM:                    Resource needs matrix for PM [n_mach, n_res]
            rho_CM:                    Resource needs matrix for CM [n_mach, n_res]
            machines_completing_maint: Machine ids whose maintenance finishes this step
                                       (their renewable resources are freed)
            rng:                       NumPy Generator

        Returns:
            (updated ResourceState, total_ordering_cost this step)
        """
        # ------------------------------------------------------------------
        # STEP 1: Consume resources for maintenance initiated this step
        # ------------------------------------------------------------------
        for m_idx, action in enumerate(maintenance_actions):
            if action == 1:  # PM
                rho = rho_PM[m_idx]   # [n_renewable + n_consumable]
                state = self._consume(state, rho)
            elif action == 2:  # CM
                rho = rho_CM[m_idx]
                state = self._consume(state, rho)

        # ------------------------------------------------------------------
        # STEP 2: Shift pipeline left — all pending orders age by 1 step
        # pending_orders[:, 0] = arriving this step (before shift)
        # After shift: pending_orders[:, 0] = what was at index 1
        # ------------------------------------------------------------------
        arriving_now = state.pending_orders[:, 0].copy()  # save before shift

        # Shift left: column k becomes column k-1
        state.pending_orders[:, :-1] = state.pending_orders[:, 1:]
        state.pending_orders[:, -1]  = 0.0  # rightmost slot now empty

        # ------------------------------------------------------------------
        # STEP 3: Deliver orders that arrived (were at lag=0 before shift)
        # ------------------------------------------------------------------
        state.consumable_inventory += arriving_now
        # No upper cap on inventory — agent learns not to over-order via cost

        # ------------------------------------------------------------------
        # STEP 4: Inject new orders from Agent 1's order_actions
        # Place at lag = L_r - 1 (rightmost valid slot for each resource)
        # Phase 3: sample actual lead time from LogNormal
        # ------------------------------------------------------------------
        ordering_cost = 0.0
        for r in range(self.n_consumable):
            qty = float(np.clip(order_actions[r], 0, self.Q_max))
            if qty <= 0:
                continue

            # Determine actual lead time for this order
            if self.stoch_level >= 3:
                actual_lead = sample_lead_time(
                    nominal_lead=float(self.nominal_lead_times[r]),
                    sigma_log=self.sigma_log_lead,
                    rng=rng,
                )
            else:
                actual_lead = int(self.nominal_lead_times[r])

            # Clip to valid range [1, max_lead_time]
            actual_lead = int(np.clip(actual_lead, 1, self.max_lead_time))

            # Place order at the correct lag position
            # lag index = actual_lead - 1
            # (lag=0 arrives next step, lag=L-1 arrives in L steps)
            lag_idx = actual_lead - 1
            state.pending_orders[r, lag_idx] += qty

            # Accumulate ordering cost
            ordering_cost += qty * self.reorder_costs[r]

        # ------------------------------------------------------------------
        # STEP 5: Free renewable resources from completed maintenance
        # ------------------------------------------------------------------
        for m_idx in machines_completing_maint:
            # We need to know what maintenance type just completed
            # This is tracked externally — for now free PM amounts
            # TODO: pass maintenance_type_completing per machine
            # For now, free rho_PM renewable amounts (conservative)
            rho = rho_PM[m_idx, :self.n_renewable]
            state.renewable_available = np.minimum(
                state.renewable_available + rho.astype(int),
                state.renewable_capacity
            )

        # Clip inventory to non-negative (should never go negative if masking works)
        state.consumable_inventory = np.maximum(state.consumable_inventory, 0.0)

        return state, ordering_cost


    def _consume(
        self,
        state: ResourceState,
        rho:   np.ndarray,   # [n_renewable + n_consumable] combined requirement vector
    ) -> ResourceState:
        """
        Deducts resource requirements for one maintenance event.
        First n_renewable entries are renewable, rest are consumable.

        Args:
            state: Current ResourceState
            rho:   Combined requirement vector [n_renewable + n_consumable]

        Returns:
            Updated ResourceState
        """
        rho_ren = rho[:self.n_renewable].astype(int)
        rho_con = rho[self.n_renewable:].astype(float)

        # Deduct from renewable available capacity
        state.renewable_available = np.maximum(
            state.renewable_available - rho_ren, 0
        )

        # Deduct from consumable inventory
        state.consumable_inventory = np.maximum(
            state.consumable_inventory - rho_con, 0.0
        )

        return state


    def compute_reorder_mask(
        self,
        state: ResourceState,
        rho_CM_max: np.ndarray,  # [n_consumable] max CM requirement across machines
    ) -> np.ndarray:
        """
        Computes action mask for Agent 1's reorder actions.
        Blocks reorder if pending pipeline already covers projected need.
        Prevents the over-ordering reward hacking degenerate policy.

        Args:
            state:       Current ResourceState
            rho_CM_max:  Max consumable units needed per CM event [n_consumable]

        Returns:
            [n_consumable] bool mask — True means reorder is ALLOWED
        """
        # Project need over lead time horizon
        net_need = state.projected_consumable_need(
            horizon=self.max_lead_time,
            rho_CM=rho_CM_max,
        )
        # Allow reorder if there's a projected need
        return net_need > 0.0


    def get_obs_dim(self) -> int:
        """Returns flat observation dimension — use in observation_spaces.py."""
        return self.n_renewable + self.n_consumable * (1 + self.max_lead_time)
