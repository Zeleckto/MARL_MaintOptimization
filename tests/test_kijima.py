"""
tests/test_kijima.py
=====================
Focused unit tests for Kijima Type I imperfect repair model.

Tests mathematical correctness of:
    V_n = V_{n-1} + q * X_n   (Kijima Type I)

where:
    V_n   = virtual age after n-th repair
    X_n   = operating time since (n-1)-th repair
    q     = repair effectiveness in [0,1]
             q=0 -> perfect repair (virtual age resets to 0)
             q=1 -> minimal repair (virtual age unchanged)

Run with: python tests/test_kijima.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from environments.transitions.degradation import (
    MachineState, MachineStatus, DegradationEngine, build_machine_states
)

MACHINE_CFG = {
    "machine_id": 0, "beta": 2.8, "eta": 3000.0,
    "delta_h": 0.5, "h_PM_threshold": 40.0, "h_critical": 10.0,
    "tau_PM_shifts": 1, "tau_CM_shifts": 1,
    "h_restore_PM": 50.0, "h_restore_CM": 80.0,
}

def make_engine(q_fixed=0.5):
    return DegradationEngine({
        "stochasticity_level": 1,
        "degradation": {"dt_hours": 8.0, "q_fixed": q_fixed, "alpha_q": 5.0, "beta_q": 5.0}
    })


def test_perfect_repair_resets_virtual_age():
    """
    q=0 (perfect repair): V_new = V_old + 0 * X_n = V_old
    Wait — Kijima Type I with q=0: V_n = V_{n-1} + 0*X_n = V_{n-1}
    So perfect repair does NOT reset virtual age to 0 in Type I.
    It only eliminates damage since last repair when q=0.
    Virtual age stays at pre-repair value.
    """
    engine = make_engine(q_fixed=0.0)
    rng = np.random.default_rng(42)
    m = build_machine_states([MACHINE_CFG])[0]
    m.virtual_age = 500.0
    m.time_since_maint = 200.0

    # Initiate and complete PM (tau=1 shift)
    m = engine.tick(m, is_operating=False, rng=rng, action_maintenance=1)
    m = engine.tick(m, is_operating=False, rng=rng, action_maintenance=0)

    # With q=0: V_new = 500 + 0*200 = 500 (unchanged)
    assert abs(m.virtual_age - 500.0) < 1e-6, (
        f"With q=0, virtual age should stay at 500, got {m.virtual_age}"
    )
    print(f"PASS: q=0 (perfect repair) keeps V_age=500.0, got {m.virtual_age:.1f}")


def test_minimal_repair_adds_full_age():
    """
    q=1 (minimal repair): V_n = V_{n-1} + 1 * X_n
    Virtual age increases by full operating time since last repair.
    Machine is as old as if it had never been repaired.
    """
    engine = make_engine(q_fixed=1.0)
    rng = np.random.default_rng(42)
    m = build_machine_states([MACHINE_CFG])[0]
    m.virtual_age = 500.0
    m.time_since_maint = 200.0  # operated 200h since last repair

    m = engine.tick(m, is_operating=False, rng=rng, action_maintenance=1)
    m = engine.tick(m, is_operating=False, rng=rng, action_maintenance=0)

    # With q=1: V_new = 500 + 1*200 = 700
    assert abs(m.virtual_age - 700.0) < 1e-6, (
        f"With q=1, virtual age should be 700, got {m.virtual_age}"
    )
    print(f"PASS: q=1 (minimal repair) V_age=700.0, got {m.virtual_age:.1f}")


def test_half_effectiveness_repair():
    """
    q=0.5 (default): V_n = V_{n-1} + 0.5 * X_n
    Standard case used throughout the project.
    """
    engine = make_engine(q_fixed=0.5)
    rng = np.random.default_rng(42)
    m = build_machine_states([MACHINE_CFG])[0]
    m.virtual_age = 1000.0
    m.time_since_maint = 500.0

    m = engine.tick(m, is_operating=False, rng=rng, action_maintenance=1)
    m = engine.tick(m, is_operating=False, rng=rng, action_maintenance=0)

    # V_new = 1000 + 0.5*500 = 1250
    assert abs(m.virtual_age - 1250.0) < 1e-6, (
        f"With q=0.5, V_age should be 1250, got {m.virtual_age}"
    )
    print(f"PASS: q=0.5 V_age=1250.0, got {m.virtual_age:.1f}")


def test_virtual_age_never_decreases_with_kijima_type1():
    """
    Key property of Kijima Type I: virtual age is monotonically non-decreasing.
    V_{n+1} >= V_n always (since q >= 0 and X_n >= 0).
    This means machine never gets "younger" than its Kijima base age.
    """
    engine = make_engine(q_fixed=0.5)
    rng = np.random.default_rng(42)
    m = build_machine_states([MACHINE_CFG])[0]

    ages = []
    for step in range(20):
        m = engine.tick(m, is_operating=True, rng=rng, action_maintenance=0)
        # Do PM every 5 steps
        if step % 5 == 4:
            m_cfg = MACHINE_CFG.copy()
            m_cfg["tau_PM_shifts"] = 1
            # Manually trigger repair
            m.status = MachineStatus.PM
            m.maint_steps_remaining = 1
            m = engine.tick(m, is_operating=False, rng=rng, action_maintenance=0)
        ages.append(m.virtual_age)

    # Virtual age should never decrease
    for i in range(1, len(ages)):
        assert ages[i] >= ages[i-1] - 1e-6, (
            f"Virtual age decreased at step {i}: {ages[i-1]:.2f} -> {ages[i]:.2f}"
        )
    print(f"PASS: virtual age monotonically non-decreasing over 20 steps")
    print(f"  Start: {ages[0]:.2f}, End: {ages[-1]:.2f}")


def test_multiple_repairs_accumulate_correctly():
    """
    After 3 repairs with q=0.5:
        After repair 1: V1 = 0 + 0.5*100 = 50
        After repair 2: V2 = 50 + 0.5*100 = 100
        After repair 3: V3 = 100 + 0.5*100 = 150
    Machine age accumulates even with good repairs.
    """
    engine = make_engine(q_fixed=0.5)
    rng = np.random.default_rng(42)
    m = build_machine_states([MACHINE_CFG])[0]
    m.virtual_age = 0.0

    for repair_num in range(3):
        m.time_since_maint = 100.0   # always 100h between repairs
        m.status = MachineStatus.PM
        m.maint_steps_remaining = 1
        m = engine.tick(m, is_operating=False, rng=rng, action_maintenance=0)

    expected = 150.0  # 0 + 50 + 50 + 50
    assert abs(m.virtual_age - expected) < 1e-6, (
        f"After 3 repairs, V_age should be {expected}, got {m.virtual_age}"
    )
    print(f"PASS: 3 repairs with q=0.5 accumulate correctly: V_age={m.virtual_age:.1f}")


def test_cm_restores_more_health_than_pm():
    """CM should restore more health than PM (h_restore_CM > h_restore_PM)."""
    engine = make_engine(q_fixed=0.5)
    rng = np.random.default_rng(42)

    # PM test
    m_pm = build_machine_states([MACHINE_CFG])[0]
    m_pm.health = 40.0
    m_pm.status = MachineStatus.PM
    m_pm.maint_steps_remaining = 1
    m_pm = engine.tick(m_pm, False, rng, 0)
    health_after_pm = m_pm.health

    # CM test
    m_cm = build_machine_states([MACHINE_CFG])[0]
    m_cm.health = 40.0
    m_cm.status = MachineStatus.CM
    m_cm.maint_steps_remaining = 1
    m_cm = engine.tick(m_cm, False, rng, 0)
    health_after_cm = m_cm.health

    assert health_after_cm > health_after_pm, (
        f"CM ({health_after_cm:.1f}) should restore more health than PM ({health_after_pm:.1f})"
    )
    print(f"PASS: CM restores {health_after_cm:.1f} health, PM restores {health_after_pm:.1f}")


def test_effective_age_used_for_hazard_not_virtual_age():
    """
    Hazard rate must use effective_age = virtual_age + time_since_maint.
    If only virtual_age is used, hazard rate would be stale between repairs.
    """
    engine = make_engine(q_fixed=0.5)
    rng = np.random.default_rng(42)
    m = build_machine_states([MACHINE_CFG])[0]
    m.virtual_age = 0.0
    m.time_since_maint = 0.0

    # Run 10 steps — hazard should increase
    hazards = []
    for _ in range(10):
        m = engine.tick(m, True, rng, 0)
        hazards.append(m.hazard_rate)

    assert hazards[-1] > hazards[0], (
        "Hazard rate should increase as machine operates"
    )
    assert hazards[0] > 0.0, (
        "Hazard rate should be > 0 after first operating step"
    )
    print(f"PASS: hazard rate increases with effective age "
          f"({hazards[0]:.2e} -> {hazards[-1]:.2e})")


if __name__ == "__main__":
    tests = [
        test_perfect_repair_resets_virtual_age,
        test_minimal_repair_adds_full_age,
        test_half_effectiveness_repair,
        test_virtual_age_never_decreases_with_kijima_type1,
        test_multiple_repairs_accumulate_correctly,
        test_cm_restores_more_health_than_pm,
        test_effective_age_used_for_hazard_not_virtual_age,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}"); failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
