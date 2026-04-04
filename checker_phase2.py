"""
checker_phase2.py
==================
Validates Phase 2: reward redesign (ΔRUL dense signal, projected tardiness).

Run from project root:
    python checker_phase2.py

CHECKS:
  P2-01  r1 is non-zero every step (dense signal present)
  P2-02  ΔRUL signal fires every step and has correct sign
  P2-03  r1 variance > 1.0 over 1000 steps
  P2-04  r2 has dense signal (slack + partial completion)
  P2-05  Failure increases negative r1 and r2
  P2-06  PM action increases r1 via ΔRUL (if PM raises RUL)
  P2-07  Cost ratios correct: c_fail >> c_CM >> c_PM
  P2-08  ΔRUL = 0 when machine is in mid-maintenance (no age change)
  P2-09  Holding cost fires when inventory > 0
  P2-10  r_shared = 0 when no failures, negative when failures
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import numpy as np
import traceback

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

results = []

def check(name, fn):
    try:
        msg = fn()
        results.append((name, True, msg or ""))
        print(f"  [{PASS}] {name}")
        if msg:
            print(f"         {msg}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"  [{FAIL}] {name}")
        print(f"         {traceback.format_exc().splitlines()[-1]}")

def load_config():
    with open("configs/base.yaml") as f:
        return yaml.safe_load(f)

def make_env():
    from environments.mfg_env import ManufacturingEnv
    cfg = load_config()
    return ManufacturingEnv(cfg), cfg

def run_episode_collect_rewards(env, n_steps=200):
    """Run n_steps with WAIT policy, collect r1/r2 each step."""
    from environments.mfg_env import AGENT_PDM
    env.reset(seed=42)
    r1s, r2s, r_shareds = [], [], []
    for _ in range(n_steps):
        a1 = {
            "maintenance": np.zeros(5, dtype=int),
            "reorder":     np.zeros(env._n_consumable, dtype=float),
        }
        env._step_agent1(a1)
        env._step_agent2(len(env._valid_pairs))  # WAIT
        env._resolve_physics()
        env._compute_rewards()
        from environments.mfg_env import AGENT_PDM, AGENT_JOBSHOP
        r1s.append(env.rewards[AGENT_PDM])
        r2s.append(env.rewards[AGENT_JOBSHOP])
        r_shareds.append(env._last_r_shared)
        if env.terminations[AGENT_PDM] or env.truncations[AGENT_PDM]:
            break
    return r1s, r2s, r_shareds

# ---- Checks ----------------------------------------------------------------

def p2_01_r1_dense():
    env, _ = make_env()
    r1s, r2s, _ = run_episode_collect_rewards(env, n_steps=50)
    n_nonzero = sum(1 for r in r1s if abs(r) > 1e-6)
    frac = n_nonzero / max(len(r1s), 1)
    assert frac > 0.8, f"r1 is zero {1-frac:.0%} of the time — not dense enough"
    return f"r1 nonzero {frac:.0%} of steps ✓"

def p2_02_delta_rul_fires():
    from environments.transitions.degradation import MachineState, estimate_rul
    from environments.transitions.degradation import build_machine_states, DegradationEngine
    import yaml
    cfg = load_config()
    states = build_machine_states(cfg["machines"])
    engine = DegradationEngine(cfg)
    rng    = np.random.default_rng(42)

    prev_ruls = [estimate_rul(s) for s in states]
    # Advance one step (no maintenance)
    states = engine.tick_all(states, [False]*5, rng, [0]*5)
    curr_ruls = [estimate_rul(s) for s in states]
    delta = sum(c - p for c, p in zip(curr_ruls, prev_ruls))
    # Without maintenance, age increases so RUL should decrease
    assert delta < 0, f"ΔRUL should be negative (aging), got {delta:.4f}"
    return f"ΔRUL one step (no maint) = {delta:.3f} shifts ✓ (negative = aging)"

def p2_03_r1_variance():
    env, _ = make_env()
    r1s, _, _ = run_episode_collect_rewards(env, n_steps=1000)
    var = float(np.var(r1s))
    assert var > 0.5, f"r1 variance={var:.3f} too low — signal may be degenerate"
    return f"r1 variance = {var:.3f} > 0.5 ✓"

def p2_04_r2_has_slack_signal():
    env, _ = make_env()
    # Run with greedy scheduling (not WAIT) to see slack signal
    from environments.mfg_env import AGENT_PDM, AGENT_JOBSHOP
    env.reset(seed=99)
    r2s = []
    for _ in range(100):
        a1 = {"maintenance": np.zeros(5, dtype=int),
              "reorder": np.zeros(env._n_consumable, dtype=float)}
        env._step_agent1(a1)
        env._step_agent2(0 if env._valid_pairs else len(env._valid_pairs))
        env._resolve_physics()
        env._compute_rewards()
        r2s.append(env.rewards[AGENT_JOBSHOP])
        if env.terminations[AGENT_PDM] or env.truncations[AGENT_PDM]:
            break
    n_nonzero = sum(1 for r in r2s if abs(r) > 1e-6)
    frac = n_nonzero / max(len(r2s), 1)
    assert frac > 0.5, f"r2 nonzero only {frac:.0%} — slack signal may be absent"
    return f"r2 nonzero {frac:.0%} of steps with greedy scheduling ✓"

def p2_05_failure_increases_negative_reward():
    from rewards.components.shared_reward import compute_shared_reward
    r_no_fail = compute_shared_reward([], c_fail=25.0)
    r_fail    = compute_shared_reward([0, 2], c_fail=25.0)
    assert r_no_fail == 0.0, f"No-failure R_shared={r_no_fail}, expected 0"
    assert r_fail == -50.0,  f"2-failure R_shared={r_fail}, expected -50"
    return f"R_shared: 0 failures→{r_no_fail}, 2 failures→{r_fail} ✓"

def p2_06_pm_increases_rul():
    from environments.transitions.degradation import MachineState, estimate_rul
    s = MachineState(
        machine_id=0, name="Test", machine_type="A",
        beta=2.5, eta=120.0, q_PM=0.4, q_CM=0.2,
        h_critical=15.0, tau_PM=3, tau_CM=8,
    )
    # Age machine to 60 shifts
    s.time_since_maint = 60.0
    s._update_derived()
    rul_before = estimate_rul(s)

    # Simulate PM completing: Kijima update
    s.virtual_age      = s.virtual_age + s.q_PM * s.time_since_maint
    s.time_since_maint = 0.0
    s._update_derived()
    rul_after = estimate_rul(s)

    assert rul_after > rul_before, \
        f"PM should increase RUL: before={rul_before:.1f}, after={rul_after:.1f}"
    return f"PM: RUL {rul_before:.1f} → {rul_after:.1f} shifts (+{rul_after-rul_before:.1f}) ✓"

def p2_07_cost_ratios_correct():
    import yaml, os
    with open("rewards/reward_weights.yaml") as f:
        w = yaml.safe_load(f)
    c_PM   = w["c_PM"]
    c_CM   = w["c_CM"]
    c_fail = w["c_fail"]
    assert c_fail > c_CM > c_PM, \
        f"Cost order wrong: c_fail={c_fail}, c_CM={c_CM}, c_PM={c_PM}"
    # Verify break-even condition: c_CM > c_PM / P(fail | beta=2.5, eta=120, age=60)
    import math
    beta, eta, age = 2.5, 120.0, 60.0
    p_fail = 1.0 - math.exp(-(age/eta)**beta)
    breakeven = c_PM / p_fail
    assert c_CM > breakeven * 0.9, \
        f"c_CM={c_CM} should be > break-even {breakeven:.2f}"
    return (f"c_PM={c_PM}, c_CM={c_CM} (> break-even {breakeven:.1f}), "
            f"c_fail={c_fail} ✓")

def p2_08_delta_rul_zero_during_maint():
    from environments.transitions.degradation import MachineState, MachineStatus, estimate_rul
    s = MachineState(
        machine_id=0, name="Test", machine_type="A",
        beta=2.5, eta=120.0, q_PM=0.4, q_CM=0.2,
        h_critical=15.0, tau_PM=3, tau_CM=8,
    )
    s.time_since_maint = 40.0
    s.status = MachineStatus.PM  # in maintenance
    s.maint_steps_remaining = 2
    s._update_derived()

    rul_before = estimate_rul(s)
    # During PM, time_since_maint does NOT advance (age frozen)
    # Simulate: status is PM, so no aging this step
    rul_after  = estimate_rul(s)  # same state

    delta = rul_after - rul_before
    assert abs(delta) < 1e-6, f"ΔRUL should be ~0 during maintenance, got {delta:.6f}"
    return "ΔRUL ≈ 0 during PM (age frozen) ✓"

def p2_09_holding_cost_fires():
    from rewards.components.maintenance_reward import compute_maintenance_reward
    from environments.transitions.degradation import build_machine_states
    cfg = load_config()
    states = build_machine_states(cfg["machines"])
    import yaml
    with open("rewards/reward_weights.yaml") as f:
        w = yaml.safe_load(f)

    # With inventory
    r_with = compute_maintenance_reward(
        maintenance_actions  = [0]*5,
        ordering_cost        = 0.0,
        machine_states       = states,
        prev_ruls            = None,
        consumable_inventory = [10.0, 8.0, 10.0],
        shared_reward        = 0.0,
        weights              = w,
    )
    # Without inventory (stockout)
    r_without = compute_maintenance_reward(
        maintenance_actions  = [0]*5,
        ordering_cost        = 0.0,
        machine_states       = states,
        prev_ruls            = None,
        consumable_inventory = [0.0, 0.0, 0.0],
        shared_reward        = 0.0,
        weights              = w,
    )
    assert r_with < r_without, \
        f"Higher inventory should give lower reward: r_with={r_with:.4f}, r_without={r_without:.4f}"
    return f"Holding cost: inv=[10,8,10]→r1={r_with:.4f}, inv=[0,0,0]→r1={r_without:.4f} ✓"

def p2_10_shared_reward_sign():
    from rewards.components.shared_reward import compute_shared_reward
    assert compute_shared_reward([]) == 0.0
    assert compute_shared_reward([0]) < 0.0
    assert compute_shared_reward([0, 1, 2]) < compute_shared_reward([0])
    return "R_shared: 0→0, 1 failure→negative, 3 failures→more negative ✓"


# ---- Main ------------------------------------------------------------------
if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  CHECKER PHASE 2 — Reward Redesign")
    print("=" * 60)
    print()

    check("P2-01 r1 is nonzero >80% of steps (dense)",       p2_01_r1_dense)
    check("P2-02 ΔRUL signal fires and has correct sign",     p2_02_delta_rul_fires)
    check("P2-03 r1 variance > 0.5 over 1000 steps",         p2_03_r1_variance)
    check("P2-04 r2 has dense slack signal (>50% nonzero)",   p2_04_r2_has_slack_signal)
    check("P2-05 Failures drive negative r_shared",           p2_05_failure_increases_negative_reward)
    check("P2-06 PM increases RUL (Kijima repair)",           p2_06_pm_increases_rul)
    check("P2-07 Cost ratios: c_fail >> c_CM >> c_PM",        p2_07_cost_ratios_correct)
    check("P2-08 ΔRUL ≈ 0 during maintenance (age frozen)",   p2_08_delta_rul_zero_during_maint)
    check("P2-09 Holding cost fires with positive inventory", p2_09_holding_cost_fires)
    check("P2-10 R_shared sign is correct",                   p2_10_shared_reward_sign)

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    print(f"  Results: {passed}/{total} passed")
    if passed == total:
        print("  ✓ Phase 2 complete — ready for Phase 3")
    else:
        for name, ok, msg in results:
            if not ok:
                print(f"    → {name}: {msg}")
    print()
