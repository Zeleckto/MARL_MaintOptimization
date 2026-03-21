"""
tests/test_inventory.py
========================
Unit tests for resource dynamics, inventory pipeline, and Markov fix.

Run with: python -m pytest tests/test_inventory.py -v
Or:       python tests/test_inventory.py  (standalone)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from environments.transitions.resource_dynamics import (
    ResourceState, ResourceManager
)


# =============================================================================
# CONFIG FIXTURE
# =============================================================================

SAMPLE_CONFIG = {
    "stochasticity_level": 1,
    "processing": {"sigma_log": 0.15},
    "resources": {
        "renewable": [
            {"resource_id": 0, "name": "Technicians",      "capacity": 3},
            {"resource_id": 1, "name": "Tools",            "capacity": 4},
            {"resource_id": 2, "name": "Maintenance Bays", "capacity": 2},
        ],
        "consumable": [
            {"resource_id": 3, "name": "Spare Parts",       "initial_inventory": 20, "lead_time_shifts": 5, "reorder_cost": 10.0},
            {"resource_id": 4, "name": "Lubricants",        "initial_inventory": 15, "lead_time_shifts": 3, "reorder_cost": 5.0},
            {"resource_id": 5, "name": "Consumable Tools",  "initial_inventory": 10, "lead_time_shifts": 4, "reorder_cost": 8.0},
        ]
    }
}

# Resource requirement matrices [n_machines=5, n_renewable+n_consumable=6]
# Rows = machines, Cols = [Techs, Tools, Bays, Parts, Lube, CTool]
RHO_PM = np.array([
    [1, 1, 1, 2, 1, 1],   # M0 CNC Mill PM
    [1, 1, 1, 2, 1, 1],   # M1 Lathe PM
    [1, 2, 1, 3, 1, 1],   # M2 Press PM
    [1, 1, 1, 2, 2, 1],   # M3 Grinder PM
    [1, 1, 1, 1, 1, 1],   # M4 Drill PM
], dtype=float)

RHO_CM = np.array([
    [2, 2, 1, 5, 2, 2],   # M0 CNC Mill CM
    [2, 2, 1, 4, 2, 2],   # M1 Lathe CM
    [2, 2, 1, 6, 2, 2],   # M2 Press CM
    [2, 2, 1, 5, 3, 2],   # M3 Grinder CM
    [1, 1, 1, 3, 1, 1],   # M4 Drill CM
], dtype=float)


def make_manager():
    return ResourceManager(SAMPLE_CONFIG)


def make_rng(seed=42):
    return np.random.default_rng(seed)


# =============================================================================
# TESTS: Reset
# =============================================================================

def test_reset_renewable_at_full_capacity():
    """After reset, all renewable resources should be at full capacity."""
    manager = make_manager()
    state = manager.reset()
    np.testing.assert_array_equal(
        state.renewable_available,
        state.renewable_capacity
    )
    print("PASS: renewable resources at full capacity after reset")


def test_reset_consumable_at_initial_inventory():
    """After reset, consumable inventory matches config initial values."""
    manager = make_manager()
    state = manager.reset()
    expected = np.array([20.0, 15.0, 10.0])
    np.testing.assert_array_almost_equal(state.consumable_inventory, expected)
    print(f"PASS: consumable inventory = {state.consumable_inventory}")


def test_reset_pending_orders_zero():
    """No orders in flight at episode start."""
    manager = make_manager()
    state = manager.reset()
    assert np.all(state.pending_orders == 0.0)
    print("PASS: pending_orders all zero after reset")


def test_max_lead_time_correct():
    """max_lead_time should be max of all lead times = 5."""
    manager = make_manager()
    assert manager.max_lead_time == 5
    print(f"PASS: max_lead_time = {manager.max_lead_time}")


# =============================================================================
# TESTS: Observation vector
# =============================================================================

def test_flat_vector_shape():
    """
    Flat obs vector: 3 renewable + 3 consumable + 3*5 pipeline = 3+3+15 = 21 dims.
    """
    manager = make_manager()
    state = manager.reset()
    vec = state.to_flat_vector()
    expected_dim = 3 + 3 + 3 * 5   # n_ren + n_con + n_con*max_lead
    assert vec.shape == (expected_dim,), f"Expected {expected_dim}, got {vec.shape}"
    assert vec.dtype == np.float32
    print(f"PASS: flat vector shape={vec.shape}, dtype={vec.dtype}")


def test_flat_vector_normalised():
    """All values in flat vector should be in [0, ~1] range."""
    manager = make_manager()
    state = manager.reset()
    vec = state.to_flat_vector()
    assert vec.min() >= 0.0, f"Negative value in obs: {vec.min()}"
    print(f"PASS: obs vector in [{vec.min():.3f}, {vec.max():.3f}]")


def test_obs_dim_matches_flat_vector():
    """obs_dim property must match actual flat vector length."""
    manager = make_manager()
    state = manager.reset()
    assert state.obs_dim == len(state.to_flat_vector())
    print(f"PASS: obs_dim={state.obs_dim} matches flat vector length")


# =============================================================================
# TESTS: Consumption
# =============================================================================

def test_pm_consumes_renewable():
    """PM action must reduce available renewable capacity."""
    manager = make_manager()
    state = manager.reset()
    rng = make_rng()

    initial_techs = state.renewable_available[0]
    state, _ = manager.step(
        state,
        maintenance_actions=[1, 0, 0, 0, 0],   # PM on machine 0
        order_actions=np.zeros(3),
        rho_PM=RHO_PM, rho_CM=RHO_CM,
        machines_completing_maint=[],
        rng=rng,
    )
    # M0 PM needs 1 technician
    assert state.renewable_available[0] < initial_techs
    print(f"PASS: PM reduces technicians {initial_techs} -> {state.renewable_available[0]}")


def test_cm_consumes_more_than_pm():
    """CM should consume more resources than PM for same machine."""
    manager = make_manager()
    rng = make_rng()

    state_pm = manager.reset()
    state_pm, _ = manager.step(
        state_pm, [1,0,0,0,0], np.zeros(3), RHO_PM, RHO_CM, [], rng
    )

    state_cm = manager.reset()
    state_cm, _ = manager.step(
        state_cm, [2,0,0,0,0], np.zeros(3), RHO_PM, RHO_CM, [], rng
    )

    # CM should leave less inventory than PM
    assert state_cm.consumable_inventory[0] < state_pm.consumable_inventory[0]
    print(f"PASS: CM consumes more parts than PM "
          f"(PM: {state_pm.consumable_inventory[0]:.1f}, "
          f"CM: {state_cm.consumable_inventory[0]:.1f})")


def test_inventory_never_goes_negative():
    """Consuming more than available should clip to 0, not go negative."""
    manager = make_manager()
    state = manager.reset()
    state.consumable_inventory = np.array([0.0, 0.0, 0.0])  # empty
    rng = make_rng()

    # Try to do CM which needs spare parts
    state, _ = manager.step(
        state, [2,0,0,0,0], np.zeros(3), RHO_PM, RHO_CM, [], rng
    )
    assert np.all(state.consumable_inventory >= 0.0)
    print("PASS: inventory clipped to 0, never negative")


# =============================================================================
# TESTS: THE MARKOV FIX — Pipeline tracking
# =============================================================================

def test_order_placed_in_pipeline():
    """Ordering 5 units of resource 0 (lead=5) should appear in pipeline."""
    manager = make_manager()
    state = manager.reset()
    rng = make_rng()

    order = np.array([5.0, 0.0, 0.0])   # order 5 spare parts
    state, cost = manager.step(
        state, [0,0,0,0,0], order, RHO_PM, RHO_CM, [], rng
    )

    # Lead time for spare parts = 5 shifts
    # After 1 step, order should be at lag index = 5-1 = 4
    assert state.pending_orders[0, 4] == 5.0, (
        f"Order should be at lag=4, pipeline: {state.pending_orders[0]}"
    )
    print(f"PASS: order placed in pipeline at lag=4: {state.pending_orders[0]}")


def test_pipeline_shifts_each_step():
    """Each step, pending orders should shift left (closer to delivery)."""
    manager = make_manager()
    state = manager.reset()
    rng = make_rng()

    # Place an order
    order = np.array([5.0, 0.0, 0.0])
    state, _ = manager.step(state, [0,0,0,0,0], order, RHO_PM, RHO_CM, [], rng)
    pipeline_after_order = state.pending_orders[0].copy()

    # Step with no new orders — pipeline should shift left
    state, _ = manager.step(
        state, [0,0,0,0,0], np.zeros(3), RHO_PM, RHO_CM, [], rng
    )
    pipeline_after_shift = state.pending_orders[0]

    # The 5 units should have moved from lag=4 to lag=3
    assert pipeline_after_shift[3] == 5.0, (
        f"After shift, order should be at lag=3: {pipeline_after_shift}"
    )
    print(f"PASS: pipeline shifts each step. After order: {pipeline_after_order}, "
          f"After shift: {pipeline_after_shift}")


def test_order_arrives_after_lead_time():
    """
    Order placed at step 0 must arrive in inventory after exactly L_r steps.
    This is the core Markov fix validation.
    """
    manager = make_manager()
    state = manager.reset()
    rng = make_rng()

    initial_parts = state.consumable_inventory[0]
    order = np.array([8.0, 0.0, 0.0])   # 8 spare parts, lead time = 5

    # Step 0: place order
    state, _ = manager.step(state, [0,0,0,0,0], order, RHO_PM, RHO_CM, [], rng)
    after_order = state.consumable_inventory[0]

    # Steps 1-4: no new orders, inventory should NOT change from order
    for _ in range(4):
        state, _ = manager.step(
            state, [0,0,0,0,0], np.zeros(3), RHO_PM, RHO_CM, [], rng
        )
    after_4_steps = state.consumable_inventory[0]

    # Step 5: order arrives
    state, _ = manager.step(
        state, [0,0,0,0,0], np.zeros(3), RHO_PM, RHO_CM, [], rng
    )
    after_5_steps = state.consumable_inventory[0]

    assert after_order == initial_parts, "Inventory should not change when order placed"
    assert after_4_steps == initial_parts, "Order should not arrive before lead time"
    assert after_5_steps == initial_parts + 8.0, (
        f"Order should arrive after 5 steps. "
        f"Expected {initial_parts + 8.0}, got {after_5_steps}"
    )
    print(f"PASS: order arrives exactly at lead time step 5 "
          f"({initial_parts} -> {after_5_steps})")


def test_pending_orders_visible_in_observation():
    """
    Pending orders must appear in the flat observation vector.
    This is the Markov fix — agent can see what's coming.
    """
    manager = make_manager()
    state = manager.reset()
    rng = make_rng()

    # Place an order
    order = np.array([5.0, 0.0, 0.0])
    state, _ = manager.step(state, [0,0,0,0,0], order, RHO_PM, RHO_CM, [], rng)

    vec = state.to_flat_vector()
    # Pipeline values are in the last n_consumable * max_lead_time dims
    # = last 15 values. With 5 units in pipeline, at least one should be > 0
    pipeline_section = vec[-(3 * 5):]
    assert pipeline_section.max() > 0.0, (
        "Pending orders not visible in observation vector — Markov fix broken"
    )
    print(f"PASS: pending orders visible in obs vector (max={pipeline_section.max():.3f})")


# =============================================================================
# TESTS: Ordering cost
# =============================================================================

def test_ordering_cost_computed_correctly():
    """Ordering 5 spare parts (cost=10 each) should return cost=50."""
    manager = make_manager()
    state = manager.reset()
    rng = make_rng()

    order = np.array([5.0, 0.0, 0.0])   # 5 spare parts @ 10.0 each
    _, cost = manager.step(state, [0,0,0,0,0], order, RHO_PM, RHO_CM, [], rng)
    assert cost == 50.0, f"Expected ordering cost 50.0, got {cost}"
    print(f"PASS: ordering cost = {cost}")


def test_no_order_zero_cost():
    """No orders placed means zero ordering cost."""
    manager = make_manager()
    state = manager.reset()
    rng = make_rng()
    _, cost = manager.step(
        state, [0,0,0,0,0], np.zeros(3), RHO_PM, RHO_CM, [], rng
    )
    assert cost == 0.0
    print("PASS: zero orders = zero cost")


# =============================================================================
# TESTS: can_do_maintenance check
# =============================================================================

def test_can_do_maintenance_true_when_resources_available():
    """With full inventory, maintenance should always be feasible."""
    manager = make_manager()
    state = manager.reset()
    rho_ren = RHO_PM[0, :3].astype(int)    # renewable needs for M0 PM
    rho_con = RHO_PM[0, 3:].astype(float)  # consumable needs for M0 PM
    assert state.can_do_maintenance(rho_ren, rho_con)
    print("PASS: can_do_maintenance=True with full resources")


def test_can_do_maintenance_false_when_empty():
    """With empty inventory, maintenance should be infeasible."""
    manager = make_manager()
    state = manager.reset()
    state.consumable_inventory = np.zeros(3)
    rho_ren = np.zeros(3, dtype=int)
    rho_con = RHO_CM[0, 3:]   # CM needs parts
    assert not state.can_do_maintenance(rho_ren, rho_con)
    print("PASS: can_do_maintenance=False with empty inventory")


# =============================================================================
# MAIN — run without pytest
# =============================================================================

if __name__ == "__main__":
    tests = [
        test_reset_renewable_at_full_capacity,
        test_reset_consumable_at_initial_inventory,
        test_reset_pending_orders_zero,
        test_max_lead_time_correct,
        test_flat_vector_shape,
        test_flat_vector_normalised,
        test_obs_dim_matches_flat_vector,
        test_pm_consumes_renewable,
        test_cm_consumes_more_than_pm,
        test_inventory_never_goes_negative,
        test_order_placed_in_pipeline,
        test_pipeline_shifts_each_step,
        test_order_arrives_after_lead_time,
        test_pending_orders_visible_in_observation,
        test_ordering_cost_computed_correctly,
        test_no_order_zero_cost,
        test_can_do_maintenance_true_when_resources_available,
        test_can_do_maintenance_false_when_empty,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
