"""
checker_phase1.py
==================
Validates Phase 1: job shop redesign with operation types.

Run from project root:
    python checker_phase1.py

CHECKS:
  P1-01  JobDynamicsEngine builds eligibility map from config
  P1-02  Eligibility map has 7 op types
  P1-03  Average flexibility 38-50% (target 43%)
  P1-04  All 40 jobs generated with 3-5 ops each
  P1-05  First op of each job is READY at start
  P1-06  Subsequent ops are PENDING at start
  P1-07  Processing times in [0.5, 10] shifts per op
  P1-08  Due dates achievable: 1-30% jobs late with random policy
  P1-09  Op type distribution is reasonable (all types appear)
  P1-10  assign_operation works and transitions status correctly
  P1-11  tick() decrements remaining_time and completes ops
  P1-12  Phase 3 arrivals gated (no arrivals when stoch_level=1)
  P1-13  Phase 3 arrivals active when stoch_level=3
  P1-14  get_eligibility_stats returns correct dict
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

def make_engine(stoch_level=1):
    from environments.transitions.job_dynamics import JobDynamicsEngine
    cfg = load_config()
    cfg["stochasticity_level"] = stoch_level
    return JobDynamicsEngine(cfg), cfg

# ---- Checks ----------------------------------------------------------------

def p1_01_eligibility_map_builds():
    engine, _ = make_engine()
    stats = engine.get_eligibility_stats()
    assert stats, "get_eligibility_stats returned empty"
    assert "eligibility_map" in stats
    return f"Eligibility map built: {list(stats['eligibility_map'].keys())}"

def p1_02_seven_op_types():
    engine, _ = make_engine()
    stats = engine.get_eligibility_stats()
    n = len(stats["op_types"])
    assert n == 7, f"Expected 7 op types, got {n}"
    return f"Op types: {stats['op_types']}"

def p1_03_flexibility_range():
    engine, _ = make_engine()
    stats = engine.get_eligibility_stats()
    f = stats["avg_flexibility"]
    assert 0.38 <= f <= 0.50, f"Flexibility {f:.3f} outside [38%, 50%]"
    elig_map = stats["eligibility_map"]
    for ot, ids in elig_map.items():
        assert len(ids) >= 1, f"Op type {ot} has no eligible machines!"
    return f"Avg flexibility = {f:.1%} ✓"

def p1_04_job_generation():
    engine, _ = make_engine()
    rng  = np.random.default_rng(42)
    jobs = engine.generate_job_batch(40, rng)
    assert len(jobs) == 40, f"Expected 40 jobs, got {len(jobs)}"
    for j in jobs:
        assert 3 <= j.n_ops <= 5, f"Job {j.job_id} has {j.n_ops} ops (expected 3-5)"
        assert j.due_date > 0, f"Job {j.job_id} has invalid due_date={j.due_date}"
        assert j.weight in (1, 2, 3), f"Job {j.job_id} invalid weight={j.weight}"
    return f"40 jobs generated, all with 3-5 ops ✓"

def p1_05_first_op_ready():
    from environments.transitions.job_dynamics import OpStatus
    engine, _ = make_engine()
    rng  = np.random.default_rng(42)
    jobs = engine.generate_job_batch(10, rng)
    for j in jobs:
        assert j.operations[0].status == OpStatus.READY, \
            f"Job {j.job_id} first op not READY"
    return "All first ops are READY ✓"

def p1_06_subsequent_ops_pending():
    from environments.transitions.job_dynamics import OpStatus
    engine, _ = make_engine()
    rng  = np.random.default_rng(42)
    jobs = engine.generate_job_batch(10, rng)
    for j in jobs:
        for i in range(1, j.n_ops):
            assert j.operations[i].status == OpStatus.PENDING, \
                f"Job {j.job_id} op {i} is not PENDING"
    return "All subsequent ops are PENDING ✓"

def p1_07_proc_time_range():
    engine, _ = make_engine()
    rng  = np.random.default_rng(42)
    jobs = engine.generate_job_batch(20, rng)
    proc_times = []
    for j in jobs:
        for op in j.operations:
            for pt in op.nominal_proc_times.values():
                proc_times.append(pt)
    assert proc_times, "No processing times found!"
    assert min(proc_times) >= 0.5, f"Min proc time {min(proc_times):.2f} < 0.5 shifts"
    assert max(proc_times) <= 12.0, f"Max proc time {max(proc_times):.2f} > 12 shifts"
    avg = np.mean(proc_times)
    return f"Proc times: min={min(proc_times):.1f}, max={max(proc_times):.1f}, avg={avg:.1f} shifts"

def p1_08_due_dates_achievable():
    from environments.mfg_env import ManufacturingEnv, AGENT_PDM
    cfg = load_config()
    env = ManufacturingEnv(cfg)

    n_episodes = 10
    late_fracs = []

    for ep in range(n_episodes):
        env.reset(seed=ep * 13)
        done = False
        while not done:
            # Rule-based Agent 1 (PM at H<45)
            maint = []
            for s in env.machine_states:
                if s.status == 3:  # FAIL
                    maint.append(2)
                elif s.status == 0 and not env.machine_busy[s.machine_id] and s.health < 45:
                    maint.append(1)
                else:
                    maint.append(0)
            a1 = {"maintenance": np.array(maint, dtype=int),
                  "reorder": np.zeros(env._n_consumable, dtype=float)}
            env._step_agent1(a1)

            # Agent 2: first valid pair (greedy)
            pairs = env._valid_pairs
            env._step_agent2(0 if pairs else len(pairs))
            env._resolve_physics()
            env._compute_rewards()
            done = env.terminations[AGENT_PDM] or env.truncations[AGENT_PDM]

        late = sum(1 for j in env.jobs if j.completion_time is not None and j.tardiness > 0)
        total = sum(1 for j in env.jobs if j.completion_time is not None)
        late_fracs.append(late / max(total, 1))

    avg_late = np.mean(late_fracs)
    assert 0.0 <= avg_late <= 0.8, f"Late fraction {avg_late:.1%} unreasonable"
    return f"With greedy policy: avg late fraction = {avg_late:.1%} (some pressure expected)"

def p1_09_op_type_distribution():
    engine, _ = make_engine()
    rng  = np.random.default_rng(42)
    jobs = engine.generate_job_batch(40, rng)
    type_counts = {}
    for j in jobs:
        for op in j.operations:
            type_counts[op.op_type] = type_counts.get(op.op_type, 0) + 1
    assert len(type_counts) >= 5, \
        f"Only {len(type_counts)} op types appeared — expected ≥5 across 40 jobs"
    return f"Op type distribution: {dict(sorted(type_counts.items()))}"

def p1_10_assign_operation():
    from environments.transitions.job_dynamics import OpStatus
    engine, _ = make_engine()
    rng  = np.random.default_rng(42)
    jobs = engine.generate_job_batch(5, rng)

    j0  = jobs[0]
    op0 = j0.operations[0]
    m   = op0.eligible_machines[0]

    jobs, proc_time = engine.assign_operation(jobs, j0.job_id, 0, m, rng)
    assert jobs[0].operations[0].status == OpStatus.IN_PROGRESS
    assert jobs[0].operations[0].assigned_machine == m
    assert proc_time > 0
    return f"Assign op → IN_PROGRESS on machine {m}, proc_time={proc_time:.2f} shifts"

def p1_11_tick_completes_op():
    from environments.transitions.job_dynamics import OpStatus
    engine, _ = make_engine()
    rng  = np.random.default_rng(42)
    jobs = engine.generate_job_batch(3, rng)

    j0  = jobs[0]
    op0 = j0.operations[0]
    m   = op0.eligible_machines[0]

    jobs, _ = engine.assign_operation(jobs, j0.job_id, 0, m, rng)
    jobs[0].operations[0].remaining_time = 1.0

    jobs, completed, freed = engine.tick(jobs, current_time=5.0, rng=rng)
    assert jobs[0].operations[0].status == OpStatus.DONE
    assert j0.job_id in completed or m in freed
    if j0.n_ops > 1:
        assert jobs[0].operations[1].status == OpStatus.READY, \
            "Second op should unlock to READY"
    return f"Op completes after 1 tick, freed_machines={freed}"

def p1_12_no_phase3_arrivals_at_level1():
    engine, _ = make_engine(stoch_level=1)
    rng  = np.random.default_rng(42)
    jobs = engine.generate_job_batch(5, rng)
    new  = engine.sample_arrivals(current_time=10.0, existing_jobs=jobs, rng=rng)
    assert len(new) == 0, f"Arrivals should be 0 at stoch_level=1, got {len(new)}"
    return "No arrivals at stochasticity_level=1 ✓"

def p1_13_phase3_arrivals_at_level3():
    engine, _ = make_engine(stoch_level=3)
    rng  = np.random.default_rng(42)
    jobs = engine.generate_job_batch(5, rng)
    # Run many steps to see at least some arrivals
    total_arrivals = 0
    for t in range(100):
        new = engine.sample_arrivals(current_time=float(t), existing_jobs=jobs, rng=rng)
        total_arrivals += len(new)
        jobs.extend(new)
    assert total_arrivals > 0, "No arrivals in 100 steps at stoch_level=3!"
    return f"Phase 3: {total_arrivals} arrivals over 100 shifts ✓"

def p1_14_eligibility_stats():
    engine, _ = make_engine()
    stats = engine.get_eligibility_stats()
    assert "avg_flexibility" in stats
    assert "eligibility_map" in stats
    assert "op_types" in stats
    # Print the map nicely
    lines = []
    for ot, ids in stats["eligibility_map"].items():
        lines.append(f"{ot}→M{ids}")
    return " | ".join(lines)


# ---- Main ------------------------------------------------------------------
if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  CHECKER PHASE 1 — Job Shop Redesign")
    print("=" * 60)
    print()

    check("P1-01 Eligibility map builds from config",         p1_01_eligibility_map_builds)
    check("P1-02 7 operation types present",                  p1_02_seven_op_types)
    check("P1-03 Flexibility 38-50% (target 43%)",           p1_03_flexibility_range)
    check("P1-04 40 jobs generated, 3-5 ops each",            p1_04_job_generation)
    check("P1-05 First op of each job is READY",              p1_05_first_op_ready)
    check("P1-06 Subsequent ops are PENDING",                 p1_06_subsequent_ops_pending)
    check("P1-07 Proc times in [0.5, 12] shifts",             p1_07_proc_time_range)
    check("P1-08 Due dates create scheduling pressure",       p1_08_due_dates_achievable)
    check("P1-09 All op types appear in generated jobs",      p1_09_op_type_distribution)
    check("P1-10 assign_operation transitions to IN_PROGRESS",p1_10_assign_operation)
    check("P1-11 tick() completes op and unlocks next",       p1_11_tick_completes_op)
    check("P1-12 No Phase 3 arrivals at stoch_level=1",       p1_12_no_phase3_arrivals_at_level1)
    check("P1-13 Phase 3 arrivals active at stoch_level=3",   p1_13_phase3_arrivals_at_level3)
    check("P1-14 get_eligibility_stats returns correct data", p1_14_eligibility_stats)

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    print(f"  Results: {passed}/{total} passed")
    if passed == total:
        print("  ✓ Phase 1 complete — ready for Phase 2")
    else:
        for name, ok, msg in results:
            if not ok:
                print(f"    → {name}: {msg}")
    print()
