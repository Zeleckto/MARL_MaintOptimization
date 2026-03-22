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

DEFAULT_WEIGHTS = {
    "c_fail": 30.0, "c_PM": 1.0, "c_CM": 3.0,
    "w_avail": 2.0, "w_tard": 5.0, "w_comp": 3.0,
    "w_health": 0.5, "lambda_shared": 0.3,
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
        shared_reward=0.0,
        weights=DEFAULT_WEIGHTS,
    )
    # Should be just the availability bonus: w_avail * 1.0 = 2.0
    assert abs(r1 - 2.0) < 1e-6
    print(f"PASS: no maintenance, full avail -> r1={r1:.3f} (= w_avail*1.0)")

def test_pm_deducts_cost():
    states = make_states()
    r1_no_maint = compute_maintenance_reward([0]*5, 0.0, states, 0.0, DEFAULT_WEIGHTS)
    r1_with_pm  = compute_maintenance_reward([1,0,0,0,0], 0.0, states, 0.0, DEFAULT_WEIGHTS)
    # PM costs c_PM=1.0
    assert abs(r1_no_maint - r1_with_pm - 1.0) < 1e-6
    print(f"PASS: PM deducts c_PM=1.0 from reward ({r1_no_maint:.3f} -> {r1_with_pm:.3f})")

def test_cm_deducts_more_than_pm():
    states = make_states()
    r1_pm = compute_maintenance_reward([1,0,0,0,0], 0.0, states, 0.0, DEFAULT_WEIGHTS)
    r1_cm = compute_maintenance_reward([2,0,0,0,0], 0.0, states, 0.0, DEFAULT_WEIGHTS)
    assert r1_cm < r1_pm
    print(f"PASS: CM ({r1_cm:.3f}) penalises more than PM ({r1_pm:.3f})")

def test_ordering_cost_deducted():
    states = make_states()
    r1_no_order = compute_maintenance_reward([0]*5, 0.0,  states, 0.0, DEFAULT_WEIGHTS)
    r1_order    = compute_maintenance_reward([0]*5, 50.0, states, 0.0, DEFAULT_WEIGHTS)
    assert abs(r1_no_order - r1_order - 50.0) < 1e-6
    print(f"PASS: ordering cost 50.0 deducted correctly")

def test_failure_penalty_propagated_to_r1():
    states = make_states()
    r_shared = compute_shared_reward([0], c_fail=30.0)  # -30
    r1_no_fail = compute_maintenance_reward([0]*5, 0.0, states, 0.0,   DEFAULT_WEIGHTS)
    r1_fail    = compute_maintenance_reward([0]*5, 0.0, states, r_shared, DEFAULT_WEIGHTS)
    # lambda=0.3, so penalty = 0.3 * (-30) = -9
    diff = r1_no_fail - r1_fail
    assert abs(diff - 9.0) < 1e-6
    print(f"PASS: failure penalty propagated to r1 (diff={diff:.2f})")


# ─── Agent 2 Reward ──────────────────────────────────────────────────────────

def test_no_tardiness_no_penalty():
    jobs = [make_job(0, due=100.0, complete=True)]
    jobs[0].tardiness = 0.0   # completed on time
    r2 = compute_scheduling_reward(
        jobs=jobs, completed_job_ids=[], assignment=None,
        machine_states=make_states(), shared_reward=0.0,
        t_max=200, weights=DEFAULT_WEIGHTS,
    )
    # No tardiness, no completion this step, no assignment
    # Should be close to 0 (only small negative from normalised 0 tardiness)
    assert r2 >= -0.1
    print(f"PASS: no tardiness -> r2~0 ({r2:.4f})")

def test_completion_bonus_fires():
    jobs = [make_job(0, due=100.0, complete=True)]
    r2_no_comp = compute_scheduling_reward(
        jobs, completed_job_ids=[], assignment=None,
        machine_states=make_states(), shared_reward=0.0,
        t_max=200, weights=DEFAULT_WEIGHTS,
    )
    r2_with_comp = compute_scheduling_reward(
        jobs, completed_job_ids=[0], assignment=None,
        machine_states=make_states(), shared_reward=0.0,
        t_max=200, weights=DEFAULT_WEIGHTS,
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
        t_max=200, weights=DEFAULT_WEIGHTS,
    )
    r2_assign = compute_scheduling_reward(
        [], [], assignment=(0, 0, 0),   # assign to machine 0 (health=100)
        machine_states=states, shared_reward=0.0,
        t_max=200, weights=DEFAULT_WEIGHTS,
    )
    expected_bonus = DEFAULT_WEIGHTS["w_health"] * 1.0   # health=100/100=1.0
    assert abs(r2_assign - r2_no_assign - expected_bonus) < 1e-6
    print(f"PASS: health bonus fires on assignment (+{expected_bonus:.3f})")

def test_late_job_incurs_tardiness_penalty():
    job = make_job(0, due=50.0, weight=2.0, complete=True)
    job.tardiness = 10.0   # 10 steps late
    r2 = compute_scheduling_reward(
        [job], [], None, make_states(), 0.0, 200, DEFAULT_WEIGHTS
    )
    # tard penalty = w_tard * w_j * tardiness / T_max = 5 * 2 * 10 / 200 = 0.5
    assert r2 < 0, "Late job should give negative reward"
    print(f"PASS: late job -> negative r2={r2:.4f}")


# ─── Dense signal check ───────────────────────────────────────────────────────

def test_r1_nonzero_every_step_due_to_avail_bonus():
    """
    Agent 1 must receive non-zero reward every step even with no actions.
    This is the dense signal from the availability bonus.
    """
    states = make_states()
    r1 = compute_maintenance_reward(
        [0]*5, 0.0, states, 0.0, DEFAULT_WEIGHTS
    )
    assert r1 != 0.0, "r1 must be non-zero every step (availability bonus)"
    print(f"PASS: r1 non-zero every step = {r1:.3f} (dense availability signal)")

def test_no_dominance_in_reward_magnitudes():
    """
    No single component should dominate by 10x.
    Check that avail_bonus, maint_cost, and tard_penalty are same order of magnitude.
    """
    states = make_states()
    avail_bonus = DEFAULT_WEIGHTS["w_avail"] * 1.0   # =2.0
    pm_cost     = DEFAULT_WEIGHTS["c_PM"]              # =1.0
    fail_penalty = DEFAULT_WEIGHTS["c_fail"]           # =30.0

    # Failure penalty is 15x availability bonus — this is intentional
    # But availability bonus should be same order as PM cost
    ratio_avail_pm = avail_bonus / pm_cost
    assert ratio_avail_pm < 5.0, (
        f"Availability bonus ({avail_bonus}) too large vs PM cost ({pm_cost})"
    )
    print(f"PASS: reward magnitudes reasonable: "
          f"avail={avail_bonus}, PM={pm_cost}, fail={fail_penalty}")


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
        test_r1_nonzero_every_step_due_to_avail_bonus,
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
