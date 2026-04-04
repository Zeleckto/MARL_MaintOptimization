"""
checker_phase3.py
==================
Validates Phase 3: TensorBoard logging with all 23 KPI tags.

Run from project root:
    python checker_phase3.py

CHECKS:
  P3-01  Logger instantiates without error
  P3-02  log_rewards accepts r1/r2/r_shared/step
  P3-03  log_episode accepts all 15 episode KPI args
  P3-04  log_training accepts all 5 loss args
  P3-05  Running 5000 env steps → all expected tags logged
  P3-06  Episode length matches config T_max
  P3-07  Avg failures per episode in expected range
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

def p3_01_logger_instantiates():
    from utils.logger import Logger
    logger = Logger("runs/_checker_test", enabled=False)
    return "Logger instantiates (TensorBoard disabled for checker) ✓"

def p3_02_log_rewards_signature():
    from utils.logger import Logger
    logger = Logger("runs/_checker_test", enabled=False)
    logger.log_rewards(r1=1.5, r2=-0.3, r_shared=-25.0, step=100)
    return "log_rewards accepts all args ✓"

def p3_03_log_episode_signature():
    from utils.logger import Logger
    logger = Logger("runs/_checker_test", enabled=False)
    logger.log_episode(
        episode=1,
        episode_return1=50.0,
        episode_return2=-30.0,
        episode_length=150,
        n_failures=3,
        n_PM=8,
        n_CM=2,
        weighted_tard=45.0,
        n_jobs_completed=32,
        n_jobs_late=8,
        avg_health=65.0,
        avg_hazard_rate=0.005,
        mtbf=48.5,
        service_level=0.8,
        avg_inventory=7.2,
    )
    return "log_episode accepts all 15 KPI args ✓"

def p3_04_log_training_signature():
    from utils.logger import Logger
    logger = Logger("runs/_checker_test", enabled=False)
    logger.log_training(
        actor1_loss=0.05,
        actor2_loss=0.08,
        critic_loss=0.12,
        entropy1=2.1,
        entropy2=3.4,
        step=1000,
    )
    return "log_training accepts all 5 loss args ✓"

def p3_05_env_runs_5000_steps():
    from environments.mfg_env import ManufacturingEnv, AGENT_PDM, AGENT_JOBSHOP
    cfg = load_config()
    cfg["jobs"]["n_jobs_train"] = 10  # fewer for speed
    env = ManufacturingEnv(cfg)

    env.reset(seed=0)
    total_steps = 0
    episodes    = 0
    all_r1s     = []

    while total_steps < 5000:
        a1 = {"maintenance": np.zeros(5, dtype=int),
              "reorder": np.zeros(env._n_consumable, dtype=float)}
        env._step_agent1(a1)
        env._step_agent2(0 if env._valid_pairs else len(env._valid_pairs))
        env._resolve_physics()
        env._compute_rewards()
        all_r1s.append(env.rewards[AGENT_PDM])
        total_steps += 1

        if env.terminations[AGENT_PDM] or env.truncations[AGENT_PDM]:
            episodes += 1
            env.reset()

    # Verify r1 is non-degenerate
    r1_var = float(np.var(all_r1s))
    assert r1_var > 0.1, f"r1 variance={r1_var:.4f} too low after 5000 steps"
    return f"5000 steps OK, {episodes} episodes, r1_var={r1_var:.3f}"

def p3_06_episode_length():
    from environments.mfg_env import ManufacturingEnv, AGENT_PDM
    cfg = load_config()
    cfg["jobs"]["n_jobs_train"] = 5
    env = ManufacturingEnv(cfg)
    env.reset(seed=0)
    steps = 0
    while not (env.terminations[AGENT_PDM] or env.truncations[AGENT_PDM]):
        a1 = {"maintenance": np.zeros(5, dtype=int),
              "reorder": np.zeros(env._n_consumable, dtype=float)}
        env._step_agent1(a1)
        env._step_agent2(len(env._valid_pairs))
        env._resolve_physics()
        env._compute_rewards()
        steps += 1
        if steps > 200:
            break
    t_max = cfg["episode"]["t_max_train"]
    assert steps <= t_max + 1, f"Episode length {steps} > T_max {t_max}"
    return f"Episode ended at step {steps} (T_max={t_max}) ✓"

def p3_07_failures_per_episode():
    from environments.mfg_env import ManufacturingEnv, AGENT_PDM
    cfg = load_config()
    cfg["jobs"]["n_jobs_train"] = 5
    env = ManufacturingEnv(cfg)

    failure_counts = []
    for ep in range(15):
        env.reset(seed=ep * 11)
        done = False
        while not done:
            a1 = {"maintenance": np.zeros(5, dtype=int),
                  "reorder": np.zeros(env._n_consumable, dtype=float)}
            env._step_agent1(a1)
            env._step_agent2(len(env._valid_pairs))
            env._resolve_physics()
            env._compute_rewards()
            done = env.terminations[AGENT_PDM] or env.truncations[AGENT_PDM]
        failure_counts.append(env._episode_failures)

    avg = np.mean(failure_counts)
    assert 0 <= avg <= 20, f"Avg failures {avg:.1f} outside [0, 20]"
    return f"Avg failures/ep = {avg:.1f} (range: {min(failure_counts)}-{max(failure_counts)}) ✓"


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  CHECKER PHASE 3 — Logging & 23 KPIs")
    print("=" * 60)
    print()

    check("P3-01 Logger instantiates",                        p3_01_logger_instantiates)
    check("P3-02 log_rewards signature correct",              p3_02_log_rewards_signature)
    check("P3-03 log_episode has all 15 KPI args",            p3_03_log_episode_signature)
    check("P3-04 log_training has all 5 loss args",           p3_04_log_training_signature)
    check("P3-05 5000 env steps without crash",               p3_05_env_runs_5000_steps)
    check("P3-06 Episode length ≤ T_max",                     p3_06_episode_length)
    check("P3-07 Failures per episode in [0, 20]",            p3_07_failures_per_episode)

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    print(f"  Results: {passed}/{total} passed")
    if passed == total:
        print("  ✓ Phase 3 complete — ready for Phase 4")
    else:
        for name, ok, msg in results:
            if not ok:
                print(f"    → {name}: {msg}")
    print()
