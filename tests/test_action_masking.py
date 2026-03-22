"""
tests/test_action_masking.py
=============================
Tests that action masking is correct for both agents.
These tests are critical — invalid actions reaching the environment
cause silent logical errors that are very hard to debug later.

Run with: python tests/test_action_masking.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from environments.transitions.degradation import (
    MachineState, MachineStatus, build_machine_states
)
from environments.transitions.resource_dynamics import ResourceManager
from environments.transitions.job_dynamics import JobDynamicsEngine, OpStatus
from environments.spaces.action_spaces import (
    build_agent1_maintenance_mask,
    build_agent1_reorder_mask,
    build_agent2_valid_actions,
    flatten_agent2_actions,
    ACTION_NONE, ACTION_PM, ACTION_CM,
)

MACHINE_CFGS = [
    {"machine_id": i, "beta": 2.8, "eta": 3000.0, "delta_h": 0.5,
     "h_PM_threshold": 40.0, "h_critical": 10.0, "tau_PM_shifts": 3,
     "tau_CM_shifts": 8, "h_restore_PM": 30.0, "h_restore_CM": 60.0}
    for i in range(5)
]

RESOURCE_CONFIG = {
    "stochasticity_level": 1,
    "processing": {"sigma_log": 0.15},
    "resources": {
        "renewable": [
            {"resource_id": 0, "name": "Technicians", "capacity": 3},
            {"resource_id": 1, "name": "Tools", "capacity": 4},
            {"resource_id": 2, "name": "Maintenance Bays", "capacity": 2},
        ],
        "consumable": [
            {"resource_id": 3, "name": "Spare Parts", "initial_inventory": 20,
             "lead_time_shifts": 5, "reorder_cost": 10.0},
            {"resource_id": 4, "name": "Lubricants", "initial_inventory": 15,
             "lead_time_shifts": 3, "reorder_cost": 5.0},
            {"resource_id": 5, "name": "Consumable Tools", "initial_inventory": 10,
             "lead_time_shifts": 4, "reorder_cost": 8.0},
        ]
    }
}

RHO_PM = np.ones((5, 6), dtype=float)
RHO_CM = np.ones((5, 6), dtype=float) * 2.0
N_RENEWABLE = 3

JOB_CONFIG = {
    "stochasticity_level": 1,
    "episode": {"dt_hours": 8.0, "t_max_train": 200},
    "jobs": {"n_ops_per_job_min": 2, "n_ops_per_job_max": 3, "lambda_arr": 0.0},
    "processing": {"sigma_log": 0.0},
    "machines": [{"machine_id": i} for i in range(5)],
}


def make_states():
    return build_machine_states(MACHINE_CFGS)

def make_resource():
    return ResourceManager(RESOURCE_CONFIG).reset()

def make_rng():
    return np.random.default_rng(42)


# ─── Agent 1 Maintenance Masking ─────────────────────────────────────────────

def test_pm_blocked_when_machine_failed():
    """Cannot PM a FAIL machine — must do CM instead."""
    states = make_states()
    states[0].status = MachineStatus.FAIL
    mask = build_agent1_maintenance_mask(
        states, [False]*5, make_resource(), RHO_PM, RHO_CM, N_RENEWABLE
    )
    assert not mask[0, ACTION_PM], "PM should be blocked for FAIL machine"
    assert mask[0, ACTION_CM],     "CM should be allowed for FAIL machine"
    print("PASS: PM blocked, CM allowed for FAIL machine")


def test_cm_blocked_when_machine_operational():
    """Cannot initiate CM on a healthy machine."""
    states = make_states()
    mask = build_agent1_maintenance_mask(
        states, [False]*5, make_resource(), RHO_PM, RHO_CM, N_RENEWABLE
    )
    assert not mask[0, ACTION_CM], "CM should be blocked for OP machine"
    assert mask[0, ACTION_PM],     "PM should be allowed for OP idle machine"
    print("PASS: CM blocked for OP machine")


def test_pm_blocked_when_machine_busy():
    """Cannot PM a machine that is currently processing a job."""
    states = make_states()
    machine_busy = [False]*5
    machine_busy[2] = True   # machine 2 is processing
    mask = build_agent1_maintenance_mask(
        states, machine_busy, make_resource(), RHO_PM, RHO_CM, N_RENEWABLE
    )
    assert not mask[2, ACTION_PM], "PM should be blocked when machine is busy"
    assert mask[0, ACTION_PM],     "PM should be allowed for idle machine"
    print("PASS: PM blocked when machine is busy")


def test_pm_blocked_when_already_under_maintenance():
    """Cannot PM a machine already doing PM."""
    states = make_states()
    states[1].status = MachineStatus.PM
    mask = build_agent1_maintenance_mask(
        states, [False]*5, make_resource(), RHO_PM, RHO_CM, N_RENEWABLE
    )
    assert not mask[1, ACTION_PM], "PM blocked when already in PM"
    assert not mask[1, ACTION_CM], "CM blocked when in PM"
    print("PASS: Both actions blocked for machine already in PM")


def test_none_action_always_allowed():
    """ACTION_NONE (do nothing) must always be in the mask."""
    states = make_states()
    # Make all machines FAIL — edge case
    for s in states:
        s.status = MachineStatus.FAIL
    mask = build_agent1_maintenance_mask(
        states, [False]*5, make_resource(), RHO_PM, RHO_CM, N_RENEWABLE
    )
    for i in range(5):
        assert mask[i, ACTION_NONE], f"NONE should always be allowed (machine {i})"
    print("PASS: ACTION_NONE always allowed for all machines")


def test_maintenance_blocked_when_no_resources():
    """PM blocked if insufficient renewable resources."""
    states = make_states()
    res = make_resource()
    res.renewable_available = np.zeros(3, dtype=int)   # no technicians/tools/bays
    mask = build_agent1_maintenance_mask(
        states, [False]*5, res, RHO_PM, RHO_CM, N_RENEWABLE
    )
    for i in range(5):
        assert not mask[i, ACTION_PM], f"PM should be blocked with no resources (machine {i})"
    print("PASS: PM blocked when renewable resources exhausted")


# ─── Agent 1 Reorder Masking ─────────────────────────────────────────────────

def test_reorder_blocked_when_pipeline_full():
    """Reorder blocked if pending pipeline already covers projected need."""
    res = make_resource()
    # Fill pipeline with large order for resource 0
    res.pending_orders[0, :] = 100.0   # 100 units in pipeline per lag slot
    rho_CM_max = RHO_CM[:, N_RENEWABLE:].max(axis=0)
    mask = build_agent1_reorder_mask(res, rho_CM_max)
    assert not mask[0], "Reorder should be blocked when pipeline already full"
    print("PASS: Reorder blocked when pipeline covers projected need")


def test_reorder_allowed_when_inventory_low():
    """Reorder allowed when inventory is low and pipeline is empty."""
    res = make_resource()
    res.consumable_inventory = np.array([0.0, 0.0, 0.0])  # empty inventory
    res.pending_orders[:] = 0.0   # empty pipeline
    rho_CM_max = RHO_CM[:, N_RENEWABLE:].max(axis=0)
    mask = build_agent1_reorder_mask(res, rho_CM_max)
    assert mask.any(), "Reorder should be allowed with empty inventory"
    print(f"PASS: Reorder allowed when inventory low: {mask}")


# ─── Agent 2 Action Masking ───────────────────────────────────────────────────

def test_agent2_cannot_assign_to_failed_machine():
    """Agent 2 must never receive a FAIL machine as a valid option."""
    rng = make_rng()
    engine = JobDynamicsEngine(JOB_CONFIG)
    jobs = engine.generate_job_batch(3, rng)

    states = make_states()
    states[0].status = MachineStatus.FAIL   # machine 0 is failed

    valid = build_agent2_valid_actions(jobs, states, [False]*5)
    pairs = flatten_agent2_actions(valid)

    for j, k, m in pairs:
        assert m != 0, f"FAIL machine 0 should not appear in valid pairs, got ({j},{k},{m})"
    print(f"PASS: FAIL machine excluded from Agent 2 valid actions ({len(pairs)} pairs)")


def test_agent2_cannot_assign_to_machine_under_pm():
    """Machine under PM must be excluded from Agent 2's actions."""
    rng = make_rng()
    engine = JobDynamicsEngine(JOB_CONFIG)
    jobs = engine.generate_job_batch(3, rng)

    states = make_states()
    states[1].status = MachineStatus.PM

    valid = build_agent2_valid_actions(jobs, states, [False]*5)
    pairs = flatten_agent2_actions(valid)

    for j, k, m in pairs:
        assert m != 1, f"PM machine 1 should not appear in valid pairs"
    print(f"PASS: PM machine excluded from Agent 2 valid actions ({len(pairs)} pairs)")


def test_agent2_cannot_assign_non_ready_operation():
    """Agent 2 can only assign READY operations."""
    rng = make_rng()
    engine = JobDynamicsEngine(JOB_CONFIG)
    jobs = engine.generate_job_batch(3, rng)

    # Manually set all ops to PENDING (undo the READY assignment)
    for job in jobs:
        for op in job.operations:
            op.status = OpStatus.PENDING

    states = make_states()
    valid = build_agent2_valid_actions(jobs, states, [False]*5)
    pairs = flatten_agent2_actions(valid)

    assert len(pairs) == 0, (
        f"No pairs should be valid when all ops are PENDING, got {len(pairs)}"
    )
    print("PASS: No valid pairs when all operations are PENDING")


def test_agent2_cannot_assign_to_busy_machine():
    """Machine currently processing another operation must be excluded."""
    rng = make_rng()
    engine = JobDynamicsEngine(JOB_CONFIG)
    jobs = engine.generate_job_batch(3, rng)

    states = make_states()
    machine_busy = [True, True, True, True, True]   # all machines busy

    valid = build_agent2_valid_actions(jobs, states, machine_busy)
    pairs = flatten_agent2_actions(valid)

    assert len(pairs) == 0, (
        f"No pairs should be valid when all machines are busy, got {len(pairs)}"
    )
    print("PASS: No valid pairs when all machines are busy")


def test_agent2_only_eligible_machines_appear():
    """Each operation can only be assigned to machines in its eligible_machines list."""
    rng = make_rng()
    engine = JobDynamicsEngine(JOB_CONFIG)
    jobs = engine.generate_job_batch(5, rng)
    states = make_states()

    valid = build_agent2_valid_actions(jobs, states, [False]*5)
    pairs = flatten_agent2_actions(valid)

    # Verify eligibility for each pair
    job_map = {j.job_id: j for j in jobs}
    for job_id, op_idx, machine_id in pairs:
        op = job_map[job_id].operations[op_idx]
        assert machine_id in op.eligible_machines, (
            f"Machine {machine_id} not eligible for op ({job_id},{op_idx})"
        )
    print(f"PASS: All {len(pairs)} valid pairs respect machine eligibility")


if __name__ == "__main__":
    tests = [
        test_pm_blocked_when_machine_failed,
        test_cm_blocked_when_machine_operational,
        test_pm_blocked_when_machine_busy,
        test_pm_blocked_when_already_under_maintenance,
        test_none_action_always_allowed,
        test_maintenance_blocked_when_no_resources,
        test_reorder_blocked_when_pipeline_full,
        test_reorder_allowed_when_inventory_low,
        test_agent2_cannot_assign_to_failed_machine,
        test_agent2_cannot_assign_to_machine_under_pm,
        test_agent2_cannot_assign_non_ready_operation,
        test_agent2_cannot_assign_to_busy_machine,
        test_agent2_only_eligible_machines_appear,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}"); failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
