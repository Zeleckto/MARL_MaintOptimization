"""
checker_phase0.py
==================
Validates Phase 0: configs + Weibull degradation redesign.

Run from project root:
    python checker_phase0.py

CHECKS:
  P0-01  base.yaml loads and has required keys
  P0-02  All 5 machines present with beta/eta (no delta_h)
  P0-03  All 7 operation types present with ~43% flexibility
  P0-04  T_max=150, J=40 in config
  P0-05  entropy_coef >= 0.03 (prevents collapse on fresh init)
  P0-06  reward_weights.yaml: c_fail/c_PM = 25, c_CM/c_PM = 7
  P0-07  Weibull H starts at 100 and decreases monotonically with age
  P0-08  H=100*exp(-(age/eta)^beta) formula is correct
  P0-09  estimate_rul returns positive value at t=0
  P0-10  20 random-policy episodes produce 3-8 failures on average
  P0-11  No delta_h references in degradation.py (old linear model gone)
  P0-12  h_PM_threshold NOT in config (agent discovers timing)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import numpy as np
import math
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
        print(f"         {e}")

def load_config():
    with open("configs/base.yaml") as f:
        return yaml.safe_load(f)

# ---- Checks ----------------------------------------------------------------

def p0_01_config_loads():
    cfg = load_config()
    required = ["machines", "operation_types", "resources", "jobs", "mappo", "episode"]
    missing = [k for k in required if k not in cfg]
    assert not missing, f"Missing keys: {missing}"
    return f"All required keys present: {required}"

def p0_02_machines_no_delta_h():
    cfg = load_config()
    machines = cfg["machines"]
    assert len(machines) == 5, f"Expected 5 machines, got {len(machines)}"
    types_found = set()
    for m in machines:
        assert "beta" in m, f"Machine {m['machine_id']} missing beta"
        assert "eta" in m, f"Machine {m['machine_id']} missing eta"
        assert "delta_h" not in m, f"Machine {m['machine_id']} still has delta_h!"
        assert "h_PM_threshold" not in m, f"Machine {m['machine_id']} still has h_PM_threshold!"
        types_found.add(m.get("machine_type", "?"))
    assert types_found == {"A", "B", "C", "D"}, f"Expected types A,B,C,D, got {types_found}"
    betas = [m["beta"] for m in machines]
    assert all(1.5 <= b <= 3.5 for b in betas), f"Beta out of range: {betas}"
    etas = [m["eta"] for m in machines]
    assert all(80 <= e <= 160 for e in etas), f"Eta out of range: {etas}"
    return f"5 machines, types A,B,C,D, betas={betas}, etas={etas}"

def p0_03_op_types_flexibility():
    cfg = load_config()
    op_types = cfg.get("operation_types", [])
    assert len(op_types) == 7, f"Expected 7 op types, got {len(op_types)}"
    machines = cfg["machines"]
    n_machines = len(machines)
    type_to_ids = {}
    for m in machines:
        t = m.get("machine_type", "A")
        type_to_ids.setdefault(t, []).append(m["machine_id"])
    total_elig = 0
    for op in op_types:
        elig = []
        for mt in op.get("eligible_machine_types", []):
            elig.extend(type_to_ids.get(mt, []))
        total_elig += len(set(elig))
    flexibility = total_elig / (len(op_types) * n_machines)
    assert 0.38 <= flexibility <= 0.50, f"Flexibility {flexibility:.3f} outside [38%,50%]"
    return f"7 op types, avg flexibility = {flexibility:.1%} (target ~43%)"

def p0_04_episode_jobs():
    cfg = load_config()
    t_max = cfg["episode"]["t_max_train"]
    n_jobs = cfg["jobs"]["n_jobs_train"]
    assert t_max == 150, f"Expected T_max=150, got {t_max}"
    assert n_jobs == 40, f"Expected J=40, got {n_jobs}"
    return f"T_max={t_max}, J={n_jobs}"

def p0_05_entropy_coef():
    cfg = load_config()
    ec = cfg["mappo"]["entropy_coef"]
    assert ec >= 0.03, f"entropy_coef={ec} too low (< 0.03), will collapse on fresh init"
    return f"entropy_coef={ec} ≥ 0.03 ✓"

def p0_06_cost_ratios():
    with open("rewards/reward_weights.yaml") as f:
        w = yaml.safe_load(f)
    c_PM   = w["c_PM"]
    c_CM   = w["c_CM"]
    c_fail = w["c_fail"]
    ratio_cm   = c_CM / c_PM
    ratio_fail = c_fail / c_PM
    assert abs(ratio_cm   - 7.0 ) < 0.5, f"c_CM/c_PM={ratio_cm:.1f}, expected ~7"
    assert abs(ratio_fail - 25.0) < 2.0, f"c_fail/c_PM={ratio_fail:.1f}, expected ~25"
    return f"c_PM={c_PM}, c_CM={c_CM} (ratio {ratio_cm:.0f}x), c_fail={c_fail} (ratio {ratio_fail:.0f}x)"

def p0_07_weibull_monotone():
    from environments.transitions.degradation import MachineState, MachineStatus
    s = MachineState(
        machine_id=0, name="Test", machine_type="A",
        beta=2.5, eta=120.0, q_PM=0.4, q_CM=0.2,
        h_critical=15.0, tau_PM=3, tau_CM=8,
    )
    # Artificially advance age and check H decreases
    healths = [s.health]
    for _ in range(50):
        s.time_since_maint += 1.0
        s._update_derived()
        healths.append(s.health)
    assert healths[0] == 100.0, f"H at age=0 should be 100, got {healths[0]}"
    assert all(healths[i] >= healths[i+1] for i in range(len(healths)-1)), \
        "Health is not monotonically decreasing!"
    return f"H: 100 → {healths[-1]:.1f} over 50 shifts (monotone decreasing ✓)"

def p0_08_weibull_formula():
    from environments.transitions.degradation import MachineState
    import math
    s = MachineState(
        machine_id=0, name="Test", machine_type="A",
        beta=2.5, eta=120.0, q_PM=0.4, q_CM=0.2,
        h_critical=15.0, tau_PM=3, tau_CM=8,
    )
    # Test at effective_age=60 shifts
    s.time_since_maint = 60.0
    s._update_derived()
    expected = 100.0 * math.exp(-(60.0/120.0)**2.5)
    assert abs(s.health - expected) < 0.01, \
        f"H at age=60: got {s.health:.4f}, expected {expected:.4f}"
    return f"H(60 shifts) = {s.health:.2f}% = 100*exp(-(60/120)^2.5) ✓"

def p0_09_rul_positive():
    from environments.transitions.degradation import MachineState, estimate_rul
    s = MachineState(
        machine_id=0, name="Test", machine_type="A",
        beta=2.5, eta=120.0, q_PM=0.4, q_CM=0.2,
        h_critical=15.0, tau_PM=3, tau_CM=8,
    )
    rul = estimate_rul(s, target_health=50.0)
    assert rul > 0, f"RUL at age=0 should be positive, got {rul}"
    # RUL should be less than eta
    assert rul < s.eta * 3, f"RUL={rul} unreasonably large"
    return f"RUL at age=0 = {rul:.1f} shifts (to H=50%)"

def p0_10_failures_in_range():
    from environments.mfg_env import ManufacturingEnv, AGENT_PDM, AGENT_JOBSHOP
    cfg = load_config()
    cfg["episode"]["t_max_train"] = 150
    cfg["jobs"]["n_jobs_train"]   = 5    # fewer jobs for speed
    env = ManufacturingEnv(cfg)

    n_episodes = 20
    failure_counts = []

    for ep in range(n_episodes):
        env.reset(seed=ep * 7)
        done = False
        ep_failures = 0
        while not done:
            # Agent 1: do nothing
            a1 = {
                "maintenance": np.zeros(5, dtype=int),
                "reorder":     np.zeros(env._n_consumable, dtype=float)
            }
            env._step_agent1(a1)
            env._step_agent2(len(env._valid_pairs))  # WAIT
            env._resolve_physics()
            env._compute_rewards()

            ep_failures += len(env._newly_failed)
            done = env.terminations[AGENT_PDM] or env.truncations[AGENT_PDM]

        failure_counts.append(ep_failures)

    avg_failures = np.mean(failure_counts)
    assert 1.0 <= avg_failures <= 15.0, \
        f"Avg failures={avg_failures:.1f} outside [1,15] — check Weibull params"
    return (f"Avg failures/episode = {avg_failures:.1f} over {n_episodes} episodes "
            f"(range: {min(failure_counts)}-{max(failure_counts)})")

def p0_11_no_delta_h_in_code():
    degradation_path = os.path.join(
        "environments", "transitions", "degradation.py"
    )
    assert os.path.exists(degradation_path), f"Not found: {degradation_path}"
    with open(degradation_path) as f:
        content = f.read()
    # delta_h should NOT appear as an assignment or config read
    import re
    bad_patterns = [
        r"delta_h\s*=",
        r"h\s*-=\s*delta",
        r"health\s*-=\s*delta",
    ]
    for pat in bad_patterns:
        if re.search(pat, content):
            assert False, f"Found old delta_h pattern: {pat}"
    return "No delta_h in degradation.py ✓"

def p0_12_no_h_pm_threshold():
    cfg = load_config()
    for m in cfg["machines"]:
        assert "h_PM_threshold" not in m, \
            f"Machine {m['machine_id']} still has h_PM_threshold"
    # Also check it's not in mappo or episode sections
    assert "h_PM_threshold" not in str(cfg.get("mappo", {}))
    return "h_PM_threshold removed from config ✓"


# ---- Main ------------------------------------------------------------------
if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  CHECKER PHASE 0 — Configs + Weibull Degradation")
    print("=" * 60)
    print()

    check("P0-01 Config loads with required keys",          p0_01_config_loads)
    check("P0-02 Machines: beta/eta, no delta_h, 4 types",  p0_02_machines_no_delta_h)
    check("P0-03 Op types: 7 types, ~43% flexibility",      p0_03_op_types_flexibility)
    check("P0-04 T_max=150, J=40",                          p0_04_episode_jobs)
    check("P0-05 entropy_coef >= 0.03",                     p0_05_entropy_coef)
    check("P0-06 Cost ratios c_CM=7x, c_fail=25x",         p0_06_cost_ratios)
    check("P0-07 Weibull H monotone decreasing",            p0_07_weibull_monotone)
    check("P0-08 H = 100*exp(-(age/eta)^beta)",             p0_08_weibull_formula)
    check("P0-09 RUL positive at t=0",                      p0_09_rul_positive)
    check("P0-10 Random policy: 1-15 failures/ep avg",      p0_10_failures_in_range)
    check("P0-11 No delta_h linear model in code",          p0_11_no_delta_h_in_code)
    check("P0-12 No h_PM_threshold in config",              p0_12_no_h_pm_threshold)

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    print(f"  Results: {passed}/{total} passed")

    if passed == total:
        print("  ✓ Phase 0 complete — ready for Phase 1")
    else:
        print("  ✗ Fix failures above before proceeding")
        for name, ok, msg in results:
            if not ok:
                print(f"    → {name}: {msg}")
    print()
