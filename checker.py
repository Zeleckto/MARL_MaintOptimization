"""
checker.py — Exhaustive pre-training environment checker
=========================================================
Verifies config, physics, env, rewards, and signal quality
before committing to a 12-hour training run.

Collects ALL failures — does not stop at first error.

Usage:
    python checker.py
    python checker.py --config configs/phase1.yaml --episodes 3
"""

import sys, os, time, argparse, traceback, copy
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Colour helpers ──────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"; X = "\033[0m"

PASS_COUNT = 0; FAIL_COUNT = 0; WARN_COUNT = 0
ISSUES = []

def ok(label, detail=""):
    global PASS_COUNT
    PASS_COUNT += 1
    tick = f"{G}PASS{X}"
    print(f"  {tick}  {label}" + (f"  [{detail}]" if detail else ""))

def fail(label, detail="", exc=None):
    global FAIL_COUNT
    FAIL_COUNT += 1
    cross = f"{R}FAIL{X}"
    tb = f"\n         {traceback.format_exc().strip()}" if exc else ""
    print(f"  {cross}  {label}" + (f"  — {detail}" if detail else "") + tb)
    ISSUES.append(("FAIL", label, detail))

def warn(label, detail=""):
    global WARN_COUNT
    WARN_COUNT += 1
    w = f"{Y}WARN{X}"
    print(f"  {w}  {label}" + (f"  — {detail}" if detail else ""))
    ISSUES.append(("WARN", label, detail))

def section(title):
    print(f"\n{B}{'═'*62}{X}")
    print(f"{B}  {title}{X}")
    print(f"{B}{'═'*62}{X}")

# ── Arg parse ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config",   default="configs/phase1.yaml")
parser.add_argument("--episodes", type=int, default=3,
                    help="Episodes for signal quality check (section H)")
args = parser.parse_args()

# ── Load config ─────────────────────────────────────────────────────────────
try:
    import yaml
    with open(os.path.join(ROOT, "configs", "base.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, args.config)) as f:
        ov = yaml.safe_load(f)
    if ov: cfg.update(ov)
except Exception as e:
    print(f"{R}FATAL: cannot load config — {e}{X}"); sys.exit(1)

# ════════════════════════════════════════════════════════════════════════════
section("A — CONFIG VALUES (Design Doc v2)")
# ════════════════════════════════════════════════════════════════════════════

def cfg_val(path, default=None):
    parts = path.split(".")
    v = cfg
    for p in parts:
        v = v.get(p, {}) if isinstance(v, dict) else None
        if v is None: return default
    return v

# Episode
t_max = cfg_val("episode.t_max_train")
if t_max == 150:   ok("episode.t_max_train = 150")
elif t_max is None: fail("episode.t_max_train", "MISSING")
else:               warn("episode.t_max_train", f"= {t_max}, design doc says 150")

# Jobs
n_jobs = cfg_val("jobs.n_jobs_train")
if n_jobs == 40:    ok("jobs.n_jobs_train = 40")
elif n_jobs is None: fail("jobs.n_jobs_train", "MISSING")
else:                warn("jobs.n_jobs_train", f"= {n_jobs}, design doc says 40")

ops_min = cfg_val("jobs.n_ops_per_job_min", 0)
ops_max = cfg_val("jobs.n_ops_per_job_max", 0)
y_avg = (ops_min + ops_max) / 2.0
if abs(y_avg - 4.0) < 0.6: ok(f"Y_avg = {y_avg:.1f}  (ops range [{ops_min},{ops_max}])")
else: warn(f"Y_avg = {y_avg:.1f}", "design doc target is 4.0")

pt_min = cfg_val("jobs.proc_time_min_hours", 0)
pt_max = cfg_val("jobs.proc_time_max_hours", 0)
if pt_min == 16.0 and pt_max == 64.0:
    ok(f"proc_time range [{pt_min},{pt_max}]h = [2,7] shifts")
else:
    warn(f"proc_time range [{pt_min},{pt_max}]h",
         "design doc says [16,56]h = [2,7] shifts")

compat = cfg_val("jobs.op_type_compatibility")
if compat:
    total = sum(len(v) for v in compat.values())
    n_types = len(compat)
    n_mach  = len(cfg.get("machines", []))
    elig_pct = total / (n_types * n_mach) * 100 if n_types and n_mach else 0
    if 40 <= elig_pct <= 46:
        ok(f"op_type eligibility = {elig_pct:.1f}%  ({total}/{n_types*n_mach} pairs)", "target 42.9%")
    else:
        warn(f"op_type eligibility = {elig_pct:.1f}%", "target 42.9%")
else:
    fail("jobs.op_type_compatibility", "MISSING — job generation will use random 60% eligibility")

# Reward weights
try:
    from rewards.reward_fn import RewardFunction
    rf = RewardFunction(cfg)
    w = rf.weights
    c_PM   = w.get("c_PM",  0)
    c_CM   = w.get("c_CM",  0)
    c_fail = w.get("c_fail",0)
    w_RUL  = w.get("w_RUL", 0)
    w_avail= w.get("w_avail",0)
    w_hold = w.get("w_hold", None)
    if c_PM==1 and c_CM==7 and c_fail==25:
        ok(f"cost ratio c_PM:c_CM:c_fail = {c_PM:.0f}:{c_CM:.0f}:{c_fail:.0f}", "1:7:25 ✓")
    else:
        warn(f"cost ratio = {c_PM:.0f}:{c_CM:.0f}:{c_fail:.0f}", "design doc says 1:7:25")
    if w_RUL <= 0.1: ok(f"w_RUL = {w_RUL}", "calibrated ✓")
    else: warn(f"w_RUL = {w_RUL}", "too high — will dominate r1")
    if w_avail <= 1.0: ok(f"w_avail = {w_avail}")
    else: warn(f"w_avail = {w_avail}", "design doc says 0.5")
    if w_hold is not None: ok(f"w_hold = {w_hold}", "holding cost active")
    else: warn("w_hold", "missing from reward_weights.yaml — no inventory holding cost")
except Exception as e:
    fail("RewardFunction load", str(e), e)

# MAPPO
ent_coef = cfg_val("mappo.entropy_coef", 0)
if ent_coef >= 0.05: ok(f"entropy_coef = {ent_coef}", ">= 0.05 ✓")
else: warn(f"entropy_coef = {ent_coef}", "< 0.05 — collapse risk")

# ════════════════════════════════════════════════════════════════════════════
section("B — WEIBULL CALIBRATION (Target: 1-2 failures/machine/episode)")
# ════════════════════════════════════════════════════════════════════════════
try:
    from scipy.special import gamma as gfn
    machines = cfg.get("machines", [])
    if not machines:
        fail("Machines list", "EMPTY — check base.yaml")
    else:
        all_ok = True
        for m in machines:
            beta    = m.get("beta", 2.5)
            eta_h   = m.get("eta", 1000.0)
            tau_cm  = m.get("tau_CM_shifts", 6)
            delta_h = m.get("delta_h", 0.5)
            h_crit  = m.get("h_critical", 10.0)

            mtbf_sh = (eta_h / 8.0) * gfn(1 + 1/beta)
            exp_fails = 150.0 / mtbf_sh
            avail   = mtbf_sh / (mtbf_sh + tau_cm)

            # Check health floor isn't hit before Weibull fires
            health_floor_sh = (100.0 - h_crit) / max(delta_h, 1e-6)

            issues = []
            if not (0.5 <= exp_fails <= 3.0):
                issues.append(f"exp_fails={exp_fails:.2f} outside [0.5,3.0]")
            if health_floor_sh < 150:
                issues.append(f"health floor at {health_floor_sh:.0f}sh < 150 "
                               f"(delta_h dominates Weibull)")

            label = f"M{m['machine_id']} {m.get('name','?'):10s}  beta={beta:.1f}  " \
                    f"eta={eta_h:.0f}h  MTBF={mtbf_sh:.0f}sh  " \
                    f"~{exp_fails:.1f} fails/ep  avail={avail:.3f}"
            if issues:
                warn(label, " | ".join(issues)); all_ok = False
            else:
                ok(label)

        fleet_mtbf = np.mean([(m.get("eta",1000)/8)*gfn(1+1/m.get("beta",2.5))
                               for m in machines])
        fleet_avail = np.mean([
            ((m.get("eta",1000)/8)*gfn(1+1/m.get("beta",2.5))) /
            ((m.get("eta",1000)/8)*gfn(1+1/m.get("beta",2.5)) + m.get("tau_CM_shifts",6))
            for m in machines])
        ok(f"Fleet avg MTBF={fleet_mtbf:.0f}sh  A={fleet_avail:.3f}")

except ImportError:
    warn("scipy not installed", "pip install scipy for Weibull checks")

# ════════════════════════════════════════════════════════════════════════════
section("C — UTILISATION (Target: 80-120%)")
# ════════════════════════════════════════════════════════════════════════════
try:
    from scipy.special import gamma as gfn
    M, T = len(cfg.get("machines",[])), cfg_val("episode.t_max_train", 150)
    J    = cfg_val("jobs.n_jobs_train", 40)
    mean_proc = ((pt_min + pt_max) / 2.0) / 8.0
    load = J * y_avg * mean_proc
    avails = [
        ((m["eta"]/8)*gfn(1+1/m["beta"])) /
        ((m["eta"]/8)*gfn(1+1/m["beta"]) + m["tau_CM_shifts"])
        for m in cfg.get("machines",[])
    ]
    A = np.mean(avails)
    cap = M * T * A
    rho = load / cap
    detail = f"load={load:.0f} / cap={cap:.0f} = {rho*100:.1f}%"
    if 0.80 <= rho <= 1.20: ok(f"Utilisation = {rho*100:.1f}%", detail)
    elif rho < 0.80:
        warn(f"Utilisation = {rho*100:.1f}%",
             "too low — scheduling too easy (increase jobs or proc times)")
    else:
        warn(f"Utilisation = {rho*100:.1f}%",
             "too high — many jobs can't complete regardless of policy")
except Exception as e:
    fail("Utilisation calc", str(e), e)

# ════════════════════════════════════════════════════════════════════════════
section("D — DEGRADATION ENGINE (Weibull + Kijima)")
# ════════════════════════════════════════════════════════════════════════════
try:
    from environments.transitions.degradation import (
        build_machine_states, DegradationEngine, MachineStatus
    )
    eng = DegradationEngine(cfg)
    states = build_machine_states(cfg["machines"])

    # Initial state
    for s in states:
        if s.health != 100.0:
            fail("Initial health", f"M{s.machine_id} health={s.health}, expected 100"); break
        if s.status != MachineStatus.OP:
            fail("Initial status", f"M{s.machine_id} status={s.status}, expected OP"); break
    else:
        ok("All machines start healthy (health=100, status=OP)")

    # Hazard rate increases with age
    rng = np.random.default_rng(0)
    s = build_machine_states([cfg["machines"][0]])[0]
    def _tick_one_d(state, is_op, action):
        if hasattr(eng, "tick"):
            return eng.tick(state, is_operating=is_op, rng=rng, action_maintenance=action)
        return eng.tick_all([state], [is_op], rng, [action])[0]
    hazards = []
    for _ in range(20):
        s = _tick_one_d(s, True, 0)
        hr = getattr(s, "hazard_rate", None)
        if hr is not None: hazards.append(hr)
    if hazards:
        if hazards[-1] > hazards[0]:
            ok(f"Hazard rate increases with age  ({hazards[0]:.2e} → {hazards[-1]:.2e})")
        else:
            warn("Hazard rate", f"not increasing ({hazards[0]:.2e} → {hazards[-1]:.2e})")

    # PM transition
    s = build_machine_states([cfg["machines"][0]])[0]
    # Use tick() if available, fall back to tick_all() for v1 compatibility
    def _tick_one(engine, state, is_operating, rng, action):
        if hasattr(engine, "tick"):
            return engine.tick(state, is_operating=is_operating,
                               rng=rng, action_maintenance=action)
        else:
            return engine.tick_all([state], [is_operating], rng, [action])[0]

    s_pm = _tick_one(eng, s, False, rng, 1)
    if s_pm.status == MachineStatus.PM:
        ok("PM transition  OP→PM  works")
    else:
        fail("PM transition", f"status={s_pm.status} after action=1, expected PM({MachineStatus.PM})")

    # CM transition (from FAIL)
    s = build_machine_states([cfg["machines"][0]])[0]
    s.status = MachineStatus.FAIL
    s_cm = _tick_one(eng, s, False, rng, 2)
    if s_cm.status == MachineStatus.CM:
        ok("CM transition  FAIL→CM  works")
    else:
        fail("CM transition", f"status={s_cm.status} after action=2 on FAIL, expected CM")

    # Health degrades when operating
    s = build_machine_states([cfg["machines"][0]])[0]
    s_after = eng.tick(s, is_operating=True, rng=rng, action_maintenance=0)
    if s_after.health < 100.0:
        ok(f"Health degrades when operating  (100 → {s_after.health:.2f})")
    else:
        warn("Health degradation", f"health={s_after.health} — not decreasing when operating")

    # Kijima: virtual age increases after repair
    s = build_machine_states([cfg["machines"][0]])[0]
    s.virtual_age = 500.0; s.time_since_maint = 200.0
    s.status = MachineStatus.PM; s.maint_steps_remaining = 1
    s_post = _tick_one_d(s, False, 0)
    if s_post.virtual_age > 500.0:
        ok(f"Kijima virtual age increases after repair  (500 → {s_post.virtual_age:.1f})")
    else:
        warn("Kijima update", f"virtual_age={s_post.virtual_age} <= 500 after repair")

except Exception as e:
    fail("Degradation engine", str(e), e)

# ════════════════════════════════════════════════════════════════════════════
section("E — ENVIRONMENT RESET & OBS QUALITY")
# ════════════════════════════════════════════════════════════════════════════
env = None
try:
    from environments.mfg_env import ManufacturingEnv, AGENT_PDM, AGENT_JOBSHOP
    env = ManufacturingEnv(cfg)
    obs_dict, _ = env.reset(seed=42)
    obs1 = obs_dict[AGENT_PDM]

    # Shape and dtype
    if obs1.ndim == 1 and obs1.dtype == np.float32:
        ok(f"obs1 shape={obs1.shape}  dtype={obs1.dtype}")
    else:
        fail("obs1 shape/dtype", f"shape={obs1.shape} dtype={obs1.dtype}")

    # No NaN/Inf
    if np.isnan(obs1).any():
        fail("obs1 NaN", f"NaN at indices: {np.where(np.isnan(obs1))[0][:5]}")
    elif np.isinf(obs1).any():
        fail("obs1 Inf", f"Inf at indices: {np.where(np.isinf(obs1))[0][:5]}")
    else:
        ok("obs1 has no NaN or Inf")

    # Values roughly normalised
    abs_max = np.abs(obs1).max()
    abs_mean = np.abs(obs1).mean()
    if abs_max > 100:
        warn(f"obs1 max value = {abs_max:.1f}", "may not be normalised — check feature vectors")
    else:
        ok(f"obs1 value range  max={abs_max:.2f}  mean={abs_mean:.3f}")

    # Job count
    n_jobs_actual = len(env.jobs)
    if n_jobs_actual == 40:
        ok(f"40 jobs generated at reset")
    else:
        warn(f"Jobs at reset = {n_jobs_actual}", f"expected 40 (check n_jobs_train)")

    # Valid pairs
    n_pairs = len(env._valid_pairs)
    if n_pairs > 0:
        ok(f"Valid action pairs at reset = {n_pairs}")
    else:
        fail("Valid action pairs", "= 0 — Agent 2 cannot act")

    # Machine statuses
    statuses = [s.status for s in env.machine_states]
    from environments.transitions.degradation import MachineStatus
    n_op = sum(1 for s in statuses if s == MachineStatus.OP)
    ok(f"Machine statuses at reset: {n_op}/5 OP  {statuses}")

    # Resource state sanity
    inv = env.resource_state.consumable_inventory
    if (inv > 0).all():
        ok(f"Consumable inventory > 0  {inv.tolist()}")
    else:
        warn(f"Some consumable inventory = 0 at reset", f"{inv.tolist()}")

except Exception as e:
    fail("Env reset", str(e), e)

# ════════════════════════════════════════════════════════════════════════════
section("F — JOB GENERATION QUALITY")
# ════════════════════════════════════════════════════════════════════════════
if env is not None:
    try:
        env.reset(seed=42)
        jobs = env.jobs

        # Proc time range
        all_proc_times = []
        for job in jobs:
            for op in job.operations:
                all_proc_times.extend(op.nominal_proc_times.values())
        if all_proc_times:
            pt_arr = np.array(all_proc_times)
            if pt_arr.min() >= 14 and pt_arr.max() <= 68:
                ok(f"Proc times in expected range  [{pt_arr.min():.0f},{pt_arr.max():.0f}]h "
                   f"(target [16,56]h)")
            else:
                warn(f"Proc times [{pt_arr.min():.0f},{pt_arr.max():.0f}]h",
                     "expected [16,64]h per design doc v2")
            ok(f"Mean proc time = {pt_arr.mean():.1f}h = {pt_arr.mean()/8:.2f} shifts  (target 4.5 shifts)")

        # Eligibility ratio
        total_ops = sum(len(j.operations) for j in jobs)
        total_eligible = sum(
            len(op.eligible_machines)
            for j in jobs for op in j.operations
        )
        n_mach = len(env.machine_states)
        elig_ratio = total_eligible / (total_ops * n_mach) if total_ops else 0
        if 0.38 <= elig_ratio <= 0.50:
            ok(f"Eligibility ratio = {elig_ratio*100:.1f}%  (target 42.9%)")
        else:
            warn(f"Eligibility ratio = {elig_ratio*100:.1f}%",
                 "target 42.9%  — check op_type_compatibility in config")

        # Due date feasibility
        infeasible = [j for j in jobs
                      if j.due_date <= j.n_ops * 2]
        if len(infeasible) == 0:
            ok(f"All {len(jobs)} jobs have feasible due dates  (due_date > n_ops×2)")
        else:
            warn(f"{len(infeasible)} jobs have tight due dates",
                 "may be infeasible — check due_date range")

        # Op count distribution
        op_counts = [len(j.operations) for j in jobs]
        ok(f"Ops per job  min={min(op_counts)}  max={max(op_counts)}  "
           f"mean={np.mean(op_counts):.1f}  (target mean=4)")

        # Guaranteed eligibility (every op has >=1 machine)
        no_machine = [j.job_id for j in jobs
                      for op in j.operations
                      if len(op.eligible_machines) == 0]
        if no_machine:
            fail("Eligibility guarantee", f"ops with 0 eligible machines in jobs {no_machine[:5]}")
        else:
            ok("Every operation has >= 1 eligible machine")

    except Exception as e:
        fail("Job generation checks", str(e), e)

# ════════════════════════════════════════════════════════════════════════════
section("G — REWARD FUNCTION SMOKE TEST")
# ════════════════════════════════════════════════════════════════════════════
if env is not None:
    try:
        env.reset(seed=0)
        r1_vals, r2_vals, rs_vals = [], [], []
        n_mach  = len(env.machine_states)
        n_consm = len(cfg["resources"]["consumable"])

        for step in range(30):
            maint   = np.zeros(n_mach, dtype=int)
            reorder = np.zeros(n_consm)
            env._step_agent1({"maintenance": maint, "reorder": reorder})
            env._step_agent2(0 if env._valid_pairs else len(env._valid_pairs))
            env._resolve_physics()
            env._compute_rewards()
            r1_vals.append(env.rewards[AGENT_PDM])
            r2_vals.append(env.rewards[AGENT_JOBSHOP])
            rs_vals.append(env._last_r_shared if hasattr(env, "_last_r_shared") else 0)

        r1 = np.array(r1_vals); r2 = np.array(r2_vals)

        # No NaN
        if np.isnan(r1).any(): fail("r1 NaN", str(np.where(np.isnan(r1))))
        elif np.isnan(r2).any(): fail("r2 NaN", str(np.where(np.isnan(r2))))
        else: ok("No NaN in 30 steps of rewards")

        # r1 has signal (not exactly constant)
        if r1.std() > 1e-6:
            ok(f"r1 has variance  mean={r1.mean():.3f}  std={r1.std():.4f}")
        else:
            fail("r1 signal", f"r1 is completely flat (std={r1.std():.2e}) — reward wiring dead")

        # r1 magnitude sanity
        if abs(r1.mean()) < 0.001:
            warn("r1 mean ≈ 0", "reward components may be cancelling — check weights")
        elif abs(r1.mean()) > 100:
            warn(f"r1 mean = {r1.mean():.1f}", "very large — check weight magnitudes")
        else:
            ok(f"r1 magnitude reasonable  mean={r1.mean():.3f}")

        # PM costs r1 (forcing PM should reduce r1 vs no PM)
        env.reset(seed=1)
        maint_none = np.zeros(n_mach, dtype=int)
        maint_pm   = np.zeros(n_mach, dtype=int); maint_pm[0] = 1

        env._step_agent1({"maintenance": maint_none, "reorder": np.zeros(n_consm)})
        env._step_agent2(len(env._valid_pairs))  # WAIT
        env._resolve_physics(); env._compute_rewards()
        r1_no_pm = env.rewards[AGENT_PDM]

        env.reset(seed=1)
        env._step_agent1({"maintenance": maint_pm, "reorder": np.zeros(n_consm)})
        env._step_agent2(len(env._valid_pairs))  # WAIT
        env._resolve_physics(); env._compute_rewards()
        r1_with_pm = env.rewards[AGENT_PDM]

        if r1_with_pm < r1_no_pm:
            ok(f"PM correctly costs r1  (no_PM={r1_no_pm:.3f} > with_PM={r1_with_pm:.3f})")
        else:
            warn(f"PM does not reduce r1",
                 f"no_PM={r1_no_pm:.3f}  with_PM={r1_with_pm:.3f} — c_PM may be 0")

    except Exception as e:
        fail("Reward smoke test", str(e), e)

# ════════════════════════════════════════════════════════════════════════════
section("H — PM/CM COUNTER VERIFICATION")
# ════════════════════════════════════════════════════════════════════════════
if env is not None:
    try:
        from environments.transitions.degradation import MachineStatus, DegradationEngine

        # ── Step 1: test PM at degradation engine level (isolated) ──
        eng2 = DegradationEngine(cfg)
        s = build_machine_states([cfg["machines"][0]])[0]
        s_before = s.status
        def _tick2(state, is_op, action):
            if hasattr(eng2, "tick"):
                return eng2.tick(state, is_operating=is_op,
                                 rng=np.random.default_rng(0), action_maintenance=action)
            return eng2.tick_all([state], [is_op], np.random.default_rng(0), [action])[0]
        s = _tick2(s, False, 1)
        if s.status == MachineStatus.PM:
            ok("Degradation engine: action=1 → PM status  (isolated test)")
        else:
            fail("Degradation engine PM",
                 f"status={s.status} after action=1, expected {MachineStatus.PM}")

        # ── Step 2: test PM counter in env (no agent2 assignment to M0) ──
        env.reset(seed=7)
        # Make sure M0 is definitely OP and not busy
        machine0_status = env.machine_states[0].status
        machine0_busy   = env.machine_busy[0]
        if machine0_status != MachineStatus.OP:
            warn("PM counter test", f"M0 status={machine0_status} at reset, skipping")
        else:
            maint = np.zeros(len(env.machine_states), dtype=int)
            maint[0] = 1
            env._step_agent1({"maintenance": maint,
                               "reorder": np.zeros(n_consm)})
            # Do NOT assign to M0: pick WAIT action
            env._step_agent2(len(env._valid_pairs))  # WAIT
            env._resolve_physics()

            if env._episode_pm >= 1:
                ok(f"env._episode_pm increments correctly  = {env._episode_pm}")
            else:
                fail("env._episode_pm",
                     f"= {env._episode_pm} after PM action on OP machine  "
                     f"(M0 was OP={machine0_status==MachineStatus.OP}, "
                     f"busy={machine0_busy})")

        # ── Step 3: test CM counter (force failure then CM) ──
        env.reset(seed=8)
        from environments.transitions.degradation import MachineStatus
        env.machine_states[0].status = MachineStatus.FAIL
        env._episode_failures = 1   # pretend a failure happened
        maint_cm = np.zeros(len(env.machine_states), dtype=int)
        maint_cm[0] = 2  # CM on M0
        env._step_agent1({"maintenance": maint_cm,
                          "reorder": np.zeros(n_consm)})
        env._step_agent2(len(env._valid_pairs))
        env._resolve_physics()
        if env._episode_cm >= 1:
            ok(f"env._episode_cm increments correctly  = {env._episode_cm}")
        else:
            fail("env._episode_cm",
                 f"= {env._episode_cm} after CM action on FAIL machine")

    except Exception as e:
        fail("PM/CM counter check", str(e), e)

# ════════════════════════════════════════════════════════════════════════════
section("I — OBSERVATION-NETWORK DIMENSION MATCH")
# ════════════════════════════════════════════════════════════════════════════
try:
    from environments.mfg_env import ManufacturingEnv, AGENT_PDM
    from agents.pdm_agent import PDMAgent
    from environments.spaces.observation_spaces import compute_agent1_obs_dim

    _env = ManufacturingEnv(cfg)
    _obs, _ = _env.reset(seed=0)
    actual_dim  = int(_obs[AGENT_PDM].shape[0])
    formula_dim = compute_agent1_obs_dim(cfg)

    if formula_dim == actual_dim:
        ok(f"Formula == actual == {actual_dim}")
    else:
        warn(f"Formula={formula_dim} actual={actual_dim}",
             "obs_dim param override will handle this, but fix observation_spaces.py")

    agent = PDMAgent(cfg, device="cpu", obs_dim=actual_dim)
    net_dim = agent.policy.trunk[0].in_features
    if net_dim == actual_dim:
        ok(f"Network input dim = {net_dim} == obs dim = {actual_dim}  MATCH ✓")
    else:
        fail(f"Network-obs mismatch",
             f"network={net_dim}  obs={actual_dim}  — shape crash on first step")

    n_params = sum(p.numel() for p in agent.parameters())
    ok(f"PDMAgent  params={n_params:,}")

except Exception as e:
    fail("Obs-network check", str(e), e)

# ════════════════════════════════════════════════════════════════════════════
section("J — ACTION MASKING")
# ════════════════════════════════════════════════════════════════════════════
if env is not None:
    try:
        env.reset(seed=42)

        # PM blocked on FAIL machine
        from environments.transitions.degradation import MachineStatus
        env.machine_states[0].status = MachineStatus.FAIL
        from environments.spaces.action_spaces import build_agent1_maintenance_mask
        n_ren = len(cfg.get("resources", {}).get("renewable", []))
        mask = build_agent1_maintenance_mask(
            env.machine_states, env.machine_busy, env.resource_state,
            env.rho_PM, env.rho_CM, n_renewable=n_ren
        )
        # For a FAIL machine: PM (action=1) should be masked (=0)
        # mask shape should be [n_machines, 3] or [n_machines*3]
        mask_arr = np.array(mask)
        if mask_arr.ndim == 2:
            pm_allowed_on_fail = mask_arr[0, 1]
        else:
            pm_allowed_on_fail = mask_arr[1]  # index 1 = PM for machine 0
        if pm_allowed_on_fail == 0:
            ok("PM correctly masked on FAIL machine")
        else:
            warn("PM mask on FAIL machine", f"mask={pm_allowed_on_fail}, expected 0")

        # Agent 2: valid pairs only include OP/available machines
        env.reset(seed=42)
        for pair in env._valid_pairs:
            _, _, m_id = pair
            if env.machine_states[m_id].status != MachineStatus.OP:
                fail("Agent2 valid pairs", f"pair {pair} uses non-OP machine {m_id}")
                break
        else:
            ok(f"All {len(env._valid_pairs)} Agent2 valid pairs use OP machines")

    except Exception as e:
        fail("Action masking check", str(e), e)

# ════════════════════════════════════════════════════════════════════════════
section("K — SIGNAL QUALITY (Mini training run)")
# ════════════════════════════════════════════════════════════════════════════
if env is not None:
    try:
        from environments.mfg_env import ManufacturingEnv, AGENT_PDM, AGENT_JOBSHOP

        n_mach  = len(env.machine_states)
        n_consm = len(cfg["resources"]["consumable"])
        ep_stats = []

        from environments.transitions.degradation import MachineStatus as _MS
        for ep in range(args.episodes):
            env.reset(seed=ep * 17)
            ep_r1 = ep_r2 = 0.0
            ep_busy_steps = np.zeros(n_mach, dtype=int)

            pm_forced_this_ep = False
            for step in range(150):  # full episode
                maint = np.zeros(n_mach, dtype=int)
                # Force PM on first available idle OP machine between steps 10-80
                # Try every step in that window until one succeeds
                if not pm_forced_this_ep and 10 <= step <= 80:
                    for mi in range(n_mach):
                        if (env.machine_states[mi].status == _MS.OP
                                and not env.machine_busy[mi]):
                            maint[mi] = 1
                            pm_forced_this_ep = True
                            break
                reorder = np.zeros(n_consm)
                ep_busy_steps += np.array(env.machine_busy, dtype=int)
                env._step_agent1({"maintenance": maint, "reorder": reorder})
                env._step_agent2(0 if env._valid_pairs else len(env._valid_pairs))
                env._resolve_physics()
                env._compute_rewards()
                ep_r1 += env.rewards[AGENT_PDM]
                ep_r2 += env.rewards[AGENT_JOBSHOP]
                if env.terminations.get(AGENT_PDM, False) or \
                   env.truncations.get(AGENT_PDM, False):
                    break

            ep_stats.append({
                "r1": ep_r1, "r2": ep_r2,
                "failures": env._episode_failures,
                "pm": env._episode_pm, "cm": env._episode_cm,
                "completions": env._episode_completions,
                "utilisation": ep_busy_steps.mean() / 150.0,
            })
            print(f"    ep{ep}  r1={ep_r1:+.1f}  r2={ep_r2:+.1f}  "
                  f"fails={env._episode_failures}  pm={env._episode_pm}  "
                  f"cm={env._episode_cm}  done={env._episode_completions}jobs")

        failures  = [e["failures"]    for e in ep_stats]
        pms       = [e["pm"]          for e in ep_stats]
        dones     = [e["completions"] for e in ep_stats]
        r1_means  = [e["r1"]          for e in ep_stats]

        # Failures happening
        if sum(failures) > 0:
            ok(f"Failures occur  total={sum(failures)} over {args.episodes} eps  "
               f"mean={np.mean(failures):.1f}/ep")
        else:
            warn("No failures in any episode",
                 "Weibull eta may be too high OR health floor not reached")

        # Jobs completing
        if sum(dones) > 0:
            ok(f"Jobs complete   total={sum(dones)}  mean={np.mean(dones):.1f}/ep")
        else:
            warn("No jobs completed",
                 "proc times may be too long OR episode too short")

        # PM counter end-to-end: we forced PM at step 30 in each episode,
        # so if counter is wired correctly, pms should be >= args.episodes
        if sum(pms) >= args.episodes:
            ok(f"PM counter end-to-end  total={sum(pms)} "
               f"(forced 1/ep × {args.episodes} eps ✓)")
        else:
            fail("PM counter end-to-end",
                 f"forced PM at step 30 each episode but _episode_pm={sum(pms)} "
                 f"(expected >={args.episodes}) — PM counter not wired correctly")

        # r1 variance (non-trivial signal)
        r1_arr = np.array(r1_means)
        if r1_arr.std() > 0.1:
            ok(f"r1 varies between episodes  std={r1_arr.std():.2f}  "
               f"mean={r1_arr.mean():.2f}")
        else:
            warn(f"r1 std={r1_arr.std():.3f} across episodes",
                 "reward may be too uniform — check cost calibration")

        # Failure rate vs Weibull prediction (adjusted for actual utilisation)
        if failures:
            try:
                from scipy.special import gamma as gfn
                pred_full = sum(
                    150.0 / ((m["eta"]/8)*gfn(1+1/m["beta"]))
                    for m in cfg.get("machines",[])
                )
                # Adjust prediction by actual machine utilisation from this run
                avg_util = np.mean([e["utilisation"] for e in ep_stats])
                pred_adj = pred_full * avg_util
                actual_rate = np.mean(failures)
                # Allow 60% slack (random agent has high variance)
                ratio = actual_rate / max(pred_adj, 0.1)
                if 0.4 <= ratio <= 2.5:
                    ok(f"Failure rate plausible  actual={actual_rate:.1f}/ep  "
                       f"pred@{avg_util*100:.0f}%util={pred_adj:.1f}/ep  "
                       f"(pred_full={pred_full:.1f}/ep@100%)")
                elif actual_rate == 0:
                    warn("No failures despite utilisation",
                         f"util={avg_util*100:.0f}%  pred={pred_adj:.1f}/ep  "
                         "— apply degradation.py fix (effective_age bug)")
                else:
                    warn(f"Failure rate outside expected range",
                         f"actual={actual_rate:.1f}/ep  "
                         f"pred@{avg_util*100:.0f}%util={pred_adj:.1f}/ep")
            except ImportError:
                pass

    except Exception as e:
        fail("Signal quality check", str(e), e)

# ════════════════════════════════════════════════════════════════════════════
section("L — TRAINING PIPELINE READINESS")
# ════════════════════════════════════════════════════════════════════════════
try:
    from training.mappo_trainer import MAPPOTrainer

    t = MAPPOTrainer(cfg)
    ok(f"MAPPOTrainer init  device={t.device}")
    ok(f"obs_dim measured from env = {t.agent1.policy.trunk[0].in_features}")

    # Fake agent2 for this check
    def _fa(obs, vp): return None, (0 if vp else len(vp)), 0.0, 1.0
    t.agent2.act = _fa
    t._estimate_value = lambda **kw: 0.0

    t._reset_env(seed=0)
    step_ok = True
    for i in range(5):
        try:
            done, trunc = t._collect_one_step()
            if done: t.episode += 1; t._reset_env()
        except Exception as e:
            fail(f"Trainer step {i}", str(e), e); step_ok = False; break
    if step_ok:
        ok(f"5 trainer steps completed  global_step={t.global_step}")

    # log_episode 15-arg call
    import numpy as np
    avg_h = float(np.mean([s.health for s in t.env.machine_states]))
    try:
        t.logger.log_episode(
            episode=0, episode_return1=1.0, episode_return2=-1.0,
            episode_length=150, n_failures=0, weighted_tard=0.1,
            n_jobs_completed=5, avg_health=avg_h,
            n_PM=t.env._episode_pm, n_CM=t.env._episode_cm,
            n_jobs_late=0, avg_hazard_rate=0.001,
            mtbf=100.0, service_level=0.8, avg_inventory=12.0,
        )
        ok("logger.log_episode (15-arg call)")
    except Exception as e:
        fail("logger.log_episode", str(e), e)
    t.logger.close()

    # Checkpoint save/load
    import tempfile
    try:
        tmp = tempfile.mkdtemp()
        from utils.checkpoint import save_checkpoint, load_checkpoint
        save_checkpoint(
            checkpoint_dir=tmp, episode=1, global_step=100,
            actor1=t.agent1.policy, actor2=t.agent2.tgin, critic=t.critic,
            optim_actor1=t.optim1, optim_actor2=t.optim2, optim_critic=t.optim_critic,
            config=cfg, tag="test",
        )
        import os as _os
        saved = _os.path.join(tmp, "test.pt")
        if _os.path.exists(saved):
            ok(f"Checkpoint save  ({_os.path.getsize(saved)//1024}KB)")
            meta = load_checkpoint(
                saved, t.agent1.policy, t.agent2.tgin, t.critic,
                device=t.device
            )
            ok(f"Checkpoint load  episode={meta['episode']}  step={meta['global_step']}")
        else:
            fail("Checkpoint save", "file not created")
    except Exception as e:
        fail("Checkpoint save/load", str(e), e)

except Exception as e:
    fail("Training pipeline", str(e), e)

# ════════════════════════════════════════════════════════════════════════════
section("SUMMARY")
# ════════════════════════════════════════════════════════════════════════════
fails = [i for i in ISSUES if i[0]=="FAIL"]
warns = [i for i in ISSUES if i[0]=="WARN"]
total = PASS_COUNT + FAIL_COUNT + WARN_COUNT

print(f"\n  Passed : {G}{PASS_COUNT}{X}")
print(f"  Warned : {Y}{WARN_COUNT}{X}")
print(f"  Failed : {R}{FAIL_COUNT}{X}")

if fails:
    print(f"\n  {R}BLOCKERS — fix before training:{X}")
    for _, label, detail in fails:
        print(f"    • {label}" + (f": {detail}" if detail else ""))

if warns:
    print(f"\n  {Y}WARNINGS — review but not blockers:{X}")
    for _, label, detail in warns:
        print(f"    • {label}" + (f": {detail}" if detail else ""))

print()
if not fails:
    print(f"  {G}NO BLOCKERS — ready to train{X}")
else:
    print(f"  {R}Fix {len(fails)} blocker(s) before starting training{X}")

print()
sys.exit(1 if fails else 0)