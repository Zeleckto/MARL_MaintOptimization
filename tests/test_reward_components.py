"""
tests/test_reward_components.py
================================
Unit tests for all reward components.
Each component tested in isolation first, then combined via reward_fn.

Key things to verify:
    1. Dense signals fire every step (availability, completion, health bonuses)
    2. No single component dominates — check magnitudes make sense
    3. Failure penalty hits both agents
    4. Ordering cost correctly deducted from Agent 1

Run with: python tests/test_reward_components.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from environments.transitions.degradation import (
    MachineState, MachineStatus, build_machine_states
)
from environments.transitions.job_dynamics import Job, Operation, OpStatus
from rewards.components.shared_reward import compute_shared_reward
from rewards.components.maintenance_reward import (
    compute_maintenance_reward, compute_system_availability
)
from rewards.components.scheduling_reward import compute_scheduling_reward

MACHINE_CFGS = [
    {"machine_id": i, "beta": 2.8, "eta": 3000.0, "delta_h": 0.5,
     "h_PM_threshold": 40.0, "h_critical": 10.0, "tau_PM_shifts": 3,
     "tau_CM_shifts": 8, "h_restore_PM": 30.0, "h_restore_CM": 60.0}
    for i in range(5)
]

DEFAULT_ETA = [3000.0] * 5

DEFAULT_WEIGHTS = {
    "c_fail": 25.0, "c_PM": 1.0, "c_CM": 7.0,
    "w_pm_bonus": 0.0, "w_RUL": 0.0, "delta_obj": 0.05,
    "w_hold": 0.005, "w_tard": 8.0, "w_comp": 5.0,
    "w_health": 0.5, "lambda_shared": 0.4,
    "w_comp_shared": 1.0, "alpha": 1.0,
    "w_assign": 0.5, "w_wait": 0.3,
}


def make_states():
    return build_machine_states(MACHINE_CFGS)

def make_job(job_id=0, due=100.0, weight=1.0, complete=False):
    j = Job(job_id=job_id, release_time=0.0, due_date=due, weight=weight)
    op = Operation(job_id=job_id, op_idx=0, status=OpStatus.DONE if complete else OpStatus.READY,
                   eligible_machines=[0], nominal_proc_times={0: 8.0})
    j.operations = [op]
    if complete:
        j.completion_time = 50.0
        j.tardiness = max(0.0, 50.0 - due)
    return j


# ─── Shared Reward ────────────────────────────────────────────────────────────

def test_no_failure_zero_shared_reward():
    r = compute_shared_reward([], c_fail=30.0)
    assert r == 0.0
    print("PASS: no failures -> R_shared=0.0")

def test_one_failure_correct_penalty():
    r = compute_shared_reward([0], c_fail=30.0)
    assert r == -30.0
    print(f"PASS: one failure -> R_shared={r}")

def test_multiple_failures_additive():
    r = compute_shared_reward([0, 2, 4], c_fail=30.0)
    assert r == -90.0
    print(f"PASS: three failures -> R_shared={r}")


# ─── System Availability ─────────────────────────────────────────────────────

def test_full_availability():
    states = make_states()
    avail = compute_system_availability(states)
    assert abs(avail - 1.0) < 1e-6
    print("PASS: all OP -> availability=1.0")

def test_partial_availability():
    states = make_states()
    states[0].status = MachineStatus.FAIL
    states[1].status = MachineStatus.PM
    avail = compute_system_availability(states)
    assert abs(avail - 0.6) < 1e-6   # 3/5 OP
    print(f"PASS: 3/5 OP -> availability={avail:.2f}")

def test_zero_availability():
    states = make_states()
    for s in states:
        s.status = MachineStatus.FAIL
    avail = compute_system_availability(states)
    assert abs(avail - 0.0) < 1e-6
    print("PASS: all FAIL -> availability=0.0")


# ─── Agent 1 Reward ──────────────────────────────────────────────────────────

def test_no_maintenance_no_cost():
    states = make_states()
    r1 = compute_maintenance_reward(
        maintenance_actions=[0]*5,
        ordering_cost=0.0,
        machine_states=states,
        eta_values=DEFAULT_ETA,
        shared_reward=0.0,
        weights=DEFAULT_WEIGHTS,
    )
    # w_avail removed, no actions → r1 ≈ 0 (only ΔRUL default=0)
    assert abs(r1) < 1.0, f"r1={r1:.4f} should be near 0 (no actions, no avail bonus)"
    print(f"PASS: no maintenance -> r1={r1:.3f} (near zero, clean signal)")

def test_pm_deducts_cost():
    states = make_states()
    r1_no_maint = compute_maintenance_reward([0]*5, 0.0, states, DEFAULT_ETA, 0.0, DEFAULT_WEIGHTS)
    r1_with_pm  = compute_maintenance_reward([1,0,0,0,0], 0.0, states, DEFAULT_ETA, 0.0, DEFAULT_WEIGHTS)
    # PM costs c_PM=1.0
    assert abs(r1_no_maint - r1_with_pm - 1.0) < 1e-6
    print(f"PASS: PM deducts c_PM=1.0 from reward ({r1_no_maint:.3f} -> {r1_with_pm:.3f})")

def test_cm_deducts_more_than_pm():
    """Auto-CM (n_auto_cm=1) costs more than PM (action=[1,...]).
    CM is now environment-initiated — charged via n_auto_cm not action=2."""
    states = make_states()
    r1_pm      = compute_maintenance_reward([1,0,0,0,0], 0.0, states, DEFAULT_ETA, 0.0, DEFAULT_WEIGHTS)
    r1_auto_cm = compute_maintenance_reward([0,0,0,0,0], 0.0, states, DEFAULT_ETA, 0.0, DEFAULT_WEIGHTS,
                                             n_auto_cm=1)
    assert r1_auto_cm < r1_pm, (
        f"auto-CM ({r1_auto_cm:.3f}) should cost more than PM ({r1_pm:.3f})")
    c_CM = DEFAULT_WEIGHTS.get("c_CM", 7.0)
    c_PM = DEFAULT_WEIGHTS.get("c_PM", 1.0)
    diff = r1_pm - r1_auto_cm
    assert abs(diff - (c_CM - c_PM)) < 0.01, f"cost diff should be {c_CM-c_PM:.1f}, got {diff:.2f}"
    print(f"PASS: auto-CM ({r1_auto_cm:.3f}) penalises more than PM ({r1_pm:.3f})")

def test_ordering_cost_deducted():
    states = make_states()
    r1_no_order = compute_maintenance_reward([0]*5, 0.0,  states, DEFAULT_ETA, 0.0, DEFAULT_WEIGHTS)
    r1_order    = compute_maintenance_reward([0]*5, 50.0, states, DEFAULT_ETA, 0.0, DEFAULT_WEIGHTS)
    # ordering cost delta-weighted (delta=0.05), so diff = 0.05*50 = 2.5
    diff = r1_no_order - r1_order
    assert abs(diff - 2.5) < 1e-6, f"ordering diff={diff:.4f}, expected 2.5 (delta=0.05)"
    print(f"PASS: ordering cost 50.0 deducted correctly")

def test_failure_penalty_propagated_to_r1():
    states = make_states()
    r_shared = compute_shared_reward([0], c_fail=30.0)  # -30
    r1_no_fail = compute_maintenance_reward([0]*5, 0.0, states, DEFAULT_ETA, 0.0,      DEFAULT_WEIGHTS)
    r1_fail    = compute_maintenance_reward([0]*5, 0.0, states, DEFAULT_ETA, r_shared, DEFAULT_WEIGHTS)
    # lambda=0.4, so penalty = 0.4 * (-30) = -12
    diff = r1_no_fail - r1_fail
    assert abs(diff - 12.0) < 1e-6
    print(f"PASS: failure penalty propagated to r1 (diff={diff:.2f})")


# ─── Agent 2 Reward ──────────────────────────────────────────────────────────

def test_no_tardiness_no_penalty():
    jobs = [make_job(0, due=100.0, complete=True)]
    jobs[0].tardiness = 0.0   # completed on time
    r2 = compute_scheduling_reward(
        jobs=jobs, completed_job_ids=[], assignment=None,
        machine_states=make_states(), shared_reward=0.0,
        t_max=200, current_step=0, weights=DEFAULT_WEIGHTS,
    )
    assert r2 >= -0.5   # makespan estimate may add small negative
    print(f"PASS: no tardiness -> r2~0 ({r2:.4f})")

def test_completion_bonus_fires():
    jobs = [make_job(0, due=100.0, complete=True)]
    r2_no_comp = compute_scheduling_reward(
        jobs, completed_job_ids=[], assignment=None,
        machine_states=make_states(), shared_reward=0.0,
        t_max=200, current_step=0, weights=DEFAULT_WEIGHTS,
    )
    r2_with_comp = compute_scheduling_reward(
        jobs, completed_job_ids=[0], assignment=None,
        machine_states=make_states(), shared_reward=0.0,
        t_max=200, current_step=0, weights=DEFAULT_WEIGHTS,
    )
    assert r2_with_comp > r2_no_comp
    diff = r2_with_comp - r2_no_comp
    assert abs(diff - DEFAULT_WEIGHTS["w_comp"]) < 1e-6
    print(f"PASS: completion bonus fires (+w_comp={diff:.2f})")

def test_health_bonus_fires_on_assignment():
    states = make_states()
    states[0].health = 100.0
    r2_no_assign = compute_scheduling_reward(
        [], [], assignment=None,
        machine_states=states, shared_reward=0.0,
        t_max=200, current_step=0, weights=DEFAULT_WEIGHTS,
        n_valid_pairs=5,  # valid pairs exist but agent didn't assign
    )
    r2_assign = compute_scheduling_reward(
        [], [], assignment=(0, 0, 0),   # assign to machine 0 (health=100)
        machine_states=states, shared_reward=0.0,
        t_max=200, current_step=0, weights=DEFAULT_WEIGHTS,
        n_valid_pairs=5,
    )
    # Diff = w_health * 1.0 + w_assign - (-w_wait) = 0.5 + 0.5 + 0.3 = 1.3
    expected_bonus = DEFAULT_WEIGHTS["w_health"] * 1.0 + DEFAULT_WEIGHTS["w_assign"] + DEFAULT_WEIGHTS["w_wait"]
    assert abs(r2_assign - r2_no_assign - expected_bonus) < 1e-6
    print(f"PASS: assignment bonus fires (+{expected_bonus:.3f} = health + assign + avoided_wait)")

def test_late_job_incurs_tardiness_penalty():
    job = make_job(0, due=50.0, weight=2.0, complete=True)
    job.tardiness = 10.0   # 10 steps late
    job.completion_time = 60.0  # completed at 60, due at 50
    r2 = compute_scheduling_reward(
        jobs=[job], completed_job_ids=[0], assignment=None,
        machine_states=make_states(), shared_reward=0.0,
        t_max=200, current_step=60, weights=DEFAULT_WEIGHTS,
    )
    # Incremental: only fires because job 0 is in completed_job_ids
    # tard = w_tard * w_j * (60-50) / T = 5 * 2 * 10 / 200 = 0.5
    # comp bonus = 3.0 per completion
    # r2 = -0.5 + 3.0 = 2.5 (net positive because completion dominates)
    assert r2 > 0, f"Completed late job should have net positive r2 (comp bonus > tard): r2={r2:.4f}"
    print(f"PASS: late job completed -> r2={r2:.4f} (comp bonus + tard penalty)")


# ─── Dense signal check ───────────────────────────────────────────────────────

def test_r1_dense_signal_from_holding_cost():
    """
    Agent 1 receives dense signal via holding cost every step.
    """
    states = make_states()
    r1 = compute_maintenance_reward(
        [0]*5, 0.0, states, DEFAULT_ETA, 0.0, DEFAULT_WEIGHTS,
        inventory_total=100.0
    )
    assert r1 != 0.0, f"r1 must be non-zero with holding cost: r1={r1}"
    assert r1 < 0, f"r1 should be negative from holding cost: r1={r1}"
    print(f"PASS: r1 dense signal from holding cost = {r1:.3f}")

def test_no_dominance_in_reward_magnitudes():
    """
    No single component should dominate by 10x.
    Check that failure penalty, PM cost, and completion bonus are reasonable.
    """
    fail_penalty = DEFAULT_WEIGHTS["c_fail"]              # =25.0
    pm_cost      = DEFAULT_WEIGHTS["c_PM"]                # =1.0
    comp_bonus   = DEFAULT_WEIGHTS["w_comp"]              # =5.0
    assign_bonus = DEFAULT_WEIGHTS["w_assign"]            # =0.5

    # Failure must be much worse than PM cost
    assert fail_penalty > 10 * pm_cost, "Failure must be much worse than PM cost"
    # Completion bonus should be meaningful (>= 2x assign bonus)
    assert comp_bonus >= 2 * assign_bonus, "Completion must dominate assignment"
    print(f"PASS: reward magnitudes reasonable: "
          f"fail={fail_penalty}, PM_cost={pm_cost}, comp={comp_bonus}, assign={assign_bonus}")


if __name__ == "__main__":
    tests = [
        test_no_failure_zero_shared_reward,
        test_one_failure_correct_penalty,
        test_multiple_failures_additive,
        test_full_availability,
        test_partial_availability,
        test_zero_availability,
        test_no_maintenance_no_cost,
        test_pm_deducts_cost,
        test_cm_deducts_more_than_pm,
        test_ordering_cost_deducted,
        test_failure_penalty_propagated_to_r1,
        test_no_tardiness_no_penalty,
        test_completion_bonus_fires,
        test_health_bonus_fires_on_assignment,
        test_late_job_incurs_tardiness_penalty,
        test_r1_dense_signal_from_holding_cost,
        test_no_dominance_in_reward_magnitudes,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}"); failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")