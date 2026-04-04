"""
sanity_check.py
================
Run this after any config or code change, before a full training run.
Gives a GO / NO-GO on 8 checks in ~30 seconds.

Usage:
    python sanity_check.py
    python sanity_check.py --config configs/phase1.yaml
"""
import argparse, sys, os, time
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; RESET = "\033[0m"

def ok(msg):  print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET}  {msg}"); return False
def warn(msg): print(f"  {YELLOW}WARN{RESET}  {msg}")

results = []

def check(name, fn):
    t0 = time.time()
    try:
        msg = fn()
        dt = time.time() - t0
        ok(f"{name}  ({dt:.2f}s)  {msg or ''}")
        results.append(True)
    except Exception as e:
        fail(f"{name}  — {e}")
        results.append(False)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/phase1.yaml")
args = parser.parse_args()

with open(os.path.join(ROOT, "configs", "base.yaml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
with open(os.path.join(ROOT, args.config)) as f:
    override = yaml.safe_load(f)
if override:
    cfg.update(override)

print(f"\n{'='*60}")
print(f"  SANITY CHECK  —  {args.config}")
print(f"{'='*60}\n")

# ── 1. Config keys ──────────────────────────────────────────────────────────
def check_config():
    needed = [
        ("episode.t_max_train",  cfg.get("episode",{}).get("t_max_train")),
        ("jobs.n_jobs_train",    cfg.get("jobs",{}).get("n_jobs_train")),
        ("mappo.entropy_coef",   cfg.get("mappo",{}).get("entropy_coef")),
        ("machines (count)",     len(cfg.get("machines", []))),
    ]
    for k, v in needed:
        assert v is not None, f"missing key: {k}"
    t_max = cfg["episode"]["t_max_train"]
    assert t_max == 150, f"t_max_train={t_max}, expected 150"
    assert cfg["jobs"]["n_jobs_train"] == 40, \
        f"n_jobs_train={cfg['jobs']['n_jobs_train']}, expected 40"
    assert cfg["mappo"]["entropy_coef"] >= 0.04, \
        f"entropy_coef={cfg['mappo']['entropy_coef']} too low (collapse risk)"
    return f"t_max={t_max}, jobs=40, entropy={cfg['mappo']['entropy_coef']}"

check("1. Config keys", check_config)

# ── 2. Reward weights ───────────────────────────────────────────────────────
def check_reward_weights():
    from rewards.reward_fn import RewardFunction
    rf = RewardFunction(cfg)
    c_fail = rf.weights.get("c_fail", 0)
    c_CM   = rf.weights.get("c_CM",   0)
    c_PM   = rf.weights.get("c_PM",   0)
    w_RUL  = rf.weights.get("w_RUL",  0)
    assert c_fail == 25.0, f"c_fail={c_fail}, expected 25"
    assert c_CM   == 7.0,  f"c_CM={c_CM}, expected 7"
    assert c_PM   == 1.0,  f"c_PM={c_PM}, expected 1"
    assert w_RUL  <= 0.1,  f"w_RUL={w_RUL} too high (will dominate r1)"
    return f"c_PM:c_CM:c_fail = {c_PM:.0f}:{c_CM:.0f}:{c_fail:.0f}  w_RUL={w_RUL}"

check("2. Reward weights (design doc ratios)", check_reward_weights)

# ── 3. Env reset + obs shapes ───────────────────────────────────────────────
def check_env_reset():
    from environments.mfg_env import ManufacturingEnv, AGENT_PDM
    env = ManufacturingEnv(cfg)
    obs_dict, _ = env.reset(seed=42)
    obs1 = obs_dict[AGENT_PDM]
    assert obs1.ndim == 1,           f"obs1 not 1D: {obs1.shape}"
    assert obs1.dtype == np.float32, f"obs1 dtype: {obs1.dtype}"
    assert len(env.jobs) == 40,      f"expected 40 jobs, got {len(env.jobs)}"
    assert len(env.machine_states) == 5, f"expected 5 machines"
    valid = env._valid_pairs
    assert len(valid) > 0, "no valid action pairs at reset"
    return f"obs1={obs1.shape}, jobs={len(env.jobs)}, valid_pairs={len(valid)}"

check("3. Env reset + obs shapes", check_env_reset)

# ── 4. Obs-network dimension match ──────────────────────────────────────────
def check_obs_network_match():
    from environments.mfg_env import ManufacturingEnv, AGENT_PDM
    from agents.pdm_agent import PDMAgent
    env = ManufacturingEnv(cfg)
    obs_dict, _ = env.reset(seed=42)
    actual_dim = obs_dict[AGENT_PDM].shape[0]
    agent = PDMAgent(cfg, device="cpu", obs_dim=actual_dim)
    network_dim = agent.policy.trunk[0].in_features
    assert actual_dim == network_dim, \
        f"obs_dim={actual_dim} != network_in={network_dim}  (shape mismatch crash)"
    return f"obs_dim={actual_dim} == network_in={network_dim}"

check("4. Obs-network dimension match", check_obs_network_match)

# ── 5. Reward signal (50-step episode) ──────────────────────────────────────
def check_reward_signal():
    from environments.mfg_env import ManufacturingEnv, AGENT_PDM, AGENT_JOBSHOP
    env = ManufacturingEnv(cfg)
    env.reset(seed=0)
    r1s, r2s = [], []
    for _ in range(50):
        maint  = np.zeros(len(env.machine_states), dtype=int)
        reorder = np.zeros(len(cfg["resources"]["consumable"]))
        env._step_agent1({"maintenance": maint, "reorder": reorder})
        env._step_agent2(0 if env._valid_pairs else len(env._valid_pairs))
        env._resolve_physics()
        env._compute_rewards()
        r1s.append(env.rewards[AGENT_PDM])
        r2s.append(env.rewards[AGENT_JOBSHOP])
    r1_std = float(np.std(r1s))
    r1_mean = float(np.mean(r1s))
    assert r1_std > 1e-6, f"r1 is completely frozen (std={r1_std:.2e}) — reward wiring dead"
    assert not np.isnan(r1s).any(), "r1 contains NaN"
    assert not np.isnan(r2s).any(), "r2 contains NaN"
    msg = f"r1 mean={r1_mean:.2f} std={r1_std:.4f}  r2 mean={np.mean(r2s):.2f}"
    if r1_std < 0.05:
        msg += "  (WARN: low variance — check reward weights if persists in training)"
    return msg

check("5. Reward signal (non-zero, no NaN)", check_reward_signal)

# ── 6. Weibull failure calibration ──────────────────────────────────────────
def check_weibull_calibration():
    from scipy.special import gamma as gfn
    machines = cfg.get("machines", [])
    fails_per_ep = []
    for m in machines:
        beta = m.get("beta", 2.5)
        eta_h = m.get("eta", 1000.0)
        eta_shifts = eta_h / 8.0
        mtbf = eta_shifts * gfn(1 + 1/beta)
        fails_per_ep.append(150.0 / mtbf)
    avg_fails = np.mean(fails_per_ep)
    assert 0.5 <= avg_fails <= 3.0, \
        f"avg expected failures/machine={avg_fails:.2f} (target 1-2)"
    detail = "  ".join(f"M{i}:{f:.1f}" for i, f in enumerate(fails_per_ep))
    return f"avg={avg_fails:.2f} fails/machine/ep  [{detail}]"

check("6. Weibull calibration (~1-2 fails/machine/ep)", check_weibull_calibration)

# ── 7. Utilisation calculation ───────────────────────────────────────────────
def check_utilisation():
    from scipy.special import gamma as gfn
    J  = cfg["jobs"]["n_jobs_train"]
    ops_min = cfg["jobs"]["n_ops_per_job_min"]
    ops_max = cfg["jobs"]["n_ops_per_job_max"]
    y_avg = (ops_min + ops_max) / 2.0
    pt_min = cfg["jobs"].get("proc_time_min_hours", 16.0)
    pt_max = cfg["jobs"].get("proc_time_max_hours", 56.0)
    mean_proc_shifts = ((pt_min + pt_max) / 2.0) / 8.0
    machines = cfg.get("machines", [])
    M = len(machines)
    T = cfg["episode"]["t_max_train"]
    avails = []
    for m in machines:
        beta = m.get("beta", 2.5)
        eta_h = m.get("eta", 1000.0)
        mtbf = (eta_h / 8.0) * gfn(1 + 1/beta)
        tau_cm = m.get("tau_CM_shifts", 6)
        avails.append(mtbf / (mtbf + tau_cm))
    A = np.mean(avails)
    load = J * y_avg * mean_proc_shifts
    cap  = M * T * A
    rho  = load / cap
    assert 0.80 <= rho <= 1.20, \
        f"utilisation={rho:.2f} outside 80-120% (scheduling decisions won't matter)"
    return f"rho={rho*100:.1f}%  (load={load:.0f} / cap={cap:.0f} machine-shifts)"

check("7. Utilisation 80-120%", check_utilisation)

# ── 8. PM/CM counter wiring ──────────────────────────────────────────────────
def check_pm_cm_counters():
    from environments.mfg_env import ManufacturingEnv, AGENT_PDM
    env = ManufacturingEnv(cfg)
    env.reset(seed=99)
    assert hasattr(env, "_episode_pm"),  "env._episode_pm missing"
    assert hasattr(env, "_episode_cm"),  "env._episode_cm missing"
    # Force a PM action on machine 0 (it's OP at reset)
    maint = np.zeros(len(env.machine_states), dtype=int)
    maint[0] = 1   # PM on M0
    reorder = np.zeros(len(cfg["resources"]["consumable"]))
    env._step_agent1({"maintenance": maint, "reorder": reorder})
    # Use WAIT (not assign) so M0 is not also assigned a job before PM check
    env._step_agent2(len(env._valid_pairs))  # WAIT action
    env._resolve_physics()
    assert env._episode_pm >= 1, \
        f"_episode_pm={env._episode_pm} after forcing PM action on OP machine"
    return f"_episode_pm={env._episode_pm}, _episode_cm={env._episode_cm}"

check("8. PM/CM counters wired", check_pm_cm_counters)

# ── Summary ─────────────────────────────────────────────────────────────────
n_pass = sum(results)
n_fail = len(results) - n_pass
print(f"\n{'='*60}")
if n_fail == 0:
    print(f"  {GREEN}ALL {n_pass}/{len(results)} CHECKS PASSED — GO for training{RESET}")
else:
    print(f"  {RED}{n_fail} FAILED  /  {n_pass} PASSED — fix before training{RESET}")
print(f"{'='*60}\n")
sys.exit(0 if n_fail == 0 else 1)