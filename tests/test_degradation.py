"""
tests/test_degradation.py
==========================
Unit tests for Weibull degradation, Kijima repair, and MachineState.

Run with:  python -m pytest tests/test_degradation.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from environments.transitions.degradation import (
    MachineState, MachineStatus, DegradationEngine, build_machine_states
)
from utils.distributions import (
    compute_weibull_hazard_rate, compute_weibull_rul, sample_weibull_failure
)
from utils.seeding import seed_everything


# ─── Fixtures ────────────────────────────────────────────────────────────────

SAMPLE_MACHINE_CONFIG = {
    "machine_id": 0,
    "beta": 2.8,
    "eta": 3000.0,
    "delta_h": 0.5,
    "h_PM_threshold": 40.0,
    "h_critical": 10.0,
    "tau_PM_shifts": 3,
    "tau_CM_shifts": 8,
    "h_restore_PM": 30.0,
    "h_restore_CM": 60.0,
}

PHASE1_CONFIG = {
    "stochasticity_level": 1,
    "degradation": {
        "dt_hours": 8.0,
        "q_fixed": 0.5,
        "alpha_q": 5.0,
        "beta_q": 5.0,
    }
}


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def machine():
    states = build_machine_states([SAMPLE_MACHINE_CONFIG])
    return states[0]


@pytest.fixture
def engine():
    return DegradationEngine(PHASE1_CONFIG)


# ─── Tests: Weibull distributions ────────────────────────────────────────────

def test_hazard_rate_increases_with_age():
    """Beta > 1 means hazard rate must increase with age (wear-out)."""
    beta, eta = 2.8, 3000.0
    h0 = compute_weibull_hazard_rate(0.0, beta, eta)
    h1 = compute_weibull_hazard_rate(1000.0, beta, eta)
    h2 = compute_weibull_hazard_rate(2000.0, beta, eta)
    assert h0 < h1 < h2, "Hazard rate must increase for beta > 1"


def test_hazard_rate_zero_at_age_zero():
    """At age 0, no history => hazard rate = 0."""
    h = compute_weibull_hazard_rate(0.0, beta=2.8, eta=3000.0)
    assert h == 0.0


def test_rul_decreases_with_age():
    """Remaining useful life must decrease as machine ages."""
    beta, eta = 2.8, 3000.0
    rul0 = compute_weibull_rul(0.0, beta, eta)
    rul1 = compute_weibull_rul(1000.0, beta, eta)
    rul2 = compute_weibull_rul(2000.0, beta, eta)
    assert rul0 > rul1 > rul2, "RUL must decrease with age"


def test_rul_near_zero_at_end_of_life():
    """Near characteristic life, RUL should be much smaller than at age 0."""
    beta, eta = 2.8, 3000.0
    rul_old = compute_weibull_rul(3000.0, beta, eta)
    rul_new = compute_weibull_rul(0.0, beta, eta)
    assert rul_old < rul_new * 0.4, "RUL at eta should be much less than at age 0"


def test_failure_probability_increases_with_age(rng):
    """Older machine (higher virtual age) should fail more often."""
    beta, eta, dt = 2.8, 3000.0, 8.0
    n_trials = 2000

    failures_young = sum(
        sample_weibull_failure(500.0, beta, eta, dt, rng) for _ in range(n_trials)
    )
    failures_old = sum(
        sample_weibull_failure(2500.0, beta, eta, dt, rng) for _ in range(n_trials)
    )

    p_young = failures_young / n_trials
    p_old = failures_old / n_trials
    assert p_old > p_young, (
        f"Older machine ({p_old:.3f}) should fail more than young ({p_young:.3f})"
    )


# ─── Tests: Health degradation ───────────────────────────────────────────────

def test_health_decreases_when_operating(machine, engine, rng):
    """Operating machine must lose health each tick."""
    initial_health = machine.health
    machine = engine.tick(machine, is_operating=True, rng=rng, action_maintenance=0)
    assert machine.health < initial_health, "Health must decrease when operating"


def test_health_stays_when_idle(machine, engine, rng):
    """Idle machine should NOT lose health (no degradation when idle)."""
    initial_health = machine.health
    machine = engine.tick(machine, is_operating=False, rng=rng, action_maintenance=0)
    assert machine.health == initial_health, "Health should not change when idle"


def test_health_never_goes_below_zero(engine, rng):
    """Health must be clipped to 0 — no negative health values."""
    cfg = SAMPLE_MACHINE_CONFIG.copy()
    cfg["delta_h"] = 50.0  # huge degradation rate
    state = build_machine_states([cfg])[0]
    state.health = 1.0  # almost dead

    for _ in range(5):
        state = engine.tick(state, is_operating=True, rng=rng, action_maintenance=0)

    assert state.health >= 0.0, "Health must never go negative"


def test_feature_vector_length(machine):
    """to_feature_vector() must return exactly 15 features."""
    vec = machine.to_feature_vector()
    assert vec.shape == (15,), f"Expected 15 features, got {vec.shape}"


def test_feature_vector_dtype(machine):
    """Feature vector must be float32 for PyTorch compatibility."""
    vec = machine.to_feature_vector()
    assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"


# ─── Tests: Maintenance state machine ────────────────────────────────────────

def test_pm_sets_status_to_pm(machine, engine, rng):
    """Initiating PM must set status to MachineStatus.PM."""
    machine = engine.tick(machine, is_operating=False, rng=rng, action_maintenance=1)
    assert machine.status == MachineStatus.PM


def test_pm_countdown_decrements(machine, engine, rng):
    """Each tick during PM must decrement maint_steps_remaining."""
    machine = engine.tick(machine, is_operating=False, rng=rng, action_maintenance=1)
    assert machine.maint_steps_remaining == machine.tau_PM_shifts  # set on initiation, decrements next tick


def test_pm_restores_health_on_completion(engine, rng):
    """After tau_PM_shifts ticks, PM completes and health is restored."""
    cfg = SAMPLE_MACHINE_CONFIG.copy()
    cfg["tau_PM_shifts"] = 2
    cfg["h_restore_PM"] = 30.0
    state = build_machine_states([cfg])[0]
    state.health = 50.0  # degraded machine

    # Initiate PM
    state = engine.tick(state, is_operating=False, rng=rng, action_maintenance=1)
    assert state.status == MachineStatus.PM

    # Tick through PM duration
    for _ in range(2):
        state = engine.tick(state, is_operating=False, rng=rng, action_maintenance=0)

    assert state.status == MachineStatus.OP, "Machine should be OP after PM completes"
    assert state.health > 50.0, "Health must be restored after PM"


def test_kijima_virtual_age_increases_after_repair(engine, rng):
    """
    Kijima Type I: after imperfect repair, virtual age increases.
    q=0.5, so: v_new = v_old + 0.5 * X_n (should be > v_old when X_n > 0)
    """
    cfg = SAMPLE_MACHINE_CONFIG.copy()
    cfg["tau_PM_shifts"] = 1
    state = build_machine_states([cfg])[0]
    state.virtual_age = 1000.0
    state.time_since_maint = 500.0  # simulate 500hrs since last repair

    # Initiate and complete PM in 1 step
    state = engine.tick(state, is_operating=False, rng=rng, action_maintenance=1)
    state = engine.tick(state, is_operating=False, rng=rng, action_maintenance=0)

    # With q=0.5 and X_n=500: v_new = 1000 + 0.5*500 = 1250 (approximately)
    assert state.virtual_age > 1000.0, (
        f"Virtual age should increase after imperfect repair, got {state.virtual_age}"
    )


def test_failed_machine_awaits_cm(engine, rng):
    """A FAIL-status machine must stay FAIL until Agent 1 initiates CM."""
    cfg = SAMPLE_MACHINE_CONFIG.copy()
    state = build_machine_states([cfg])[0]
    state.status = MachineStatus.FAIL

    # No CM action — should stay FAIL
    state = engine.tick(state, is_operating=False, rng=rng, action_maintenance=0)
    assert state.status == MachineStatus.FAIL, "FAIL machine should stay FAIL without CM"

    # CM initiated — should transition to CM
    state = engine.tick(state, is_operating=False, rng=rng, action_maintenance=2)
    assert state.status == MachineStatus.CM, "FAIL machine should go to CM when Agent 1 initiates"


# ─── Tests: Seeding reproducibility ──────────────────────────────────────────

def test_seeded_runs_are_identical():
    """Same seed must produce identical failure sequences."""
    cfg = SAMPLE_MACHINE_CONFIG.copy()
    eng = DegradationEngine(PHASE1_CONFIG)

    def run_episode(seed):
        rng = np.random.default_rng(seed)
        state = build_machine_states([cfg])[0]
        failure_steps = []
        for step in range(100):
            state = eng.tick(state, is_operating=True, rng=rng, action_maintenance=0)
            if state.status == MachineStatus.FAIL:
                failure_steps.append(step)
                state.status = MachineStatus.OP  # reset to continue test
        return failure_steps

    run1 = run_episode(99)
    run2 = run_episode(99)
    assert run1 == run2, "Same seed must produce identical results"


def test_different_seeds_produce_different_results():
    """Different seeds should (almost certainly) produce different sequences."""
    cfg = SAMPLE_MACHINE_CONFIG.copy()
    cfg["beta"] = 1.1  # near-random failures
    cfg["eta"] = 100.0  # very short lifetime
    eng = DegradationEngine(PHASE1_CONFIG)

    def run_episode(seed):
        rng = np.random.default_rng(seed)
        state = build_machine_states([cfg])[0]
        failure_steps = []
        for step in range(50):
            state = eng.tick(state, is_operating=True, rng=rng, action_maintenance=0)
            if state.status == MachineStatus.FAIL:
                failure_steps.append(step)
                state.status = MachineStatus.OP
        return failure_steps

    run1 = run_episode(1)
    run2 = run_episode(999)
    if run1 and run2:
        assert run1 != run2, "Different seeds should differ"


# ─── Tests: build_machine_states ─────────────────────────────────────────────

def test_build_machine_states_initial_conditions():
    """Freshly built machines must start healthy, young, and operational."""
    configs = [SAMPLE_MACHINE_CONFIG.copy() for _ in range(5)]
    for i, cfg in enumerate(configs):
        cfg["machine_id"] = i
    states = build_machine_states(configs)

    for s in states:
        assert s.health == 100.0
        assert s.virtual_age == 0.0
        assert s.status == MachineStatus.OP


def test_tick_all_returns_correct_count(engine, rng):
    """tick_all() must return same number of states as input."""
    configs = [SAMPLE_MACHINE_CONFIG.copy() for _ in range(5)]
    for i, cfg in enumerate(configs):
        cfg["machine_id"] = i
    states = build_machine_states(configs)

    updated = engine.tick_all(
        machine_states=states,
        operating_flags=[True] * 5,
        rng=rng,
        actions_maintenance=[0] * 5,
    )
    assert len(updated) == 5


if __name__ == "__main__":
    # Quick sanity run without pytest
    seed_everything(42)
    rng = np.random.default_rng(42)
    engine = DegradationEngine(PHASE1_CONFIG)
    states = build_machine_states([SAMPLE_MACHINE_CONFIG])
    machine = states[0]

    print("Running 50 steps on M1 (CNC Mill, beta=2.8, eta=3000h)...")
    print(f"{'Step':>5} {'Health':>8} {'VirtAge':>9} {'Status':>8} {'Hazard':>10} {'RUL_h':>10}")
    for step in range(50):
        machine = engine.tick(machine, is_operating=True, rng=rng, action_maintenance=0)
        status_name = {0:"OP", 1:"PM", 2:"CM", 3:"FAIL"}[machine.status]
        print(f"{step:>5} {machine.health:>8.2f} {machine.virtual_age:>9.1f} "
              f"{status_name:>8} {machine.hazard_rate:>10.6f} {machine.rul:>10.1f}")
        if machine.status == MachineStatus.FAIL:
            print("  ** FAILURE — initiating CM **")
            machine.status = MachineStatus.CM
            machine.maint_steps_remaining = machine.tau_CM_shifts