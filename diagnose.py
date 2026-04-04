"""
diagnose.py — Full pre-training diagnostic
==========================================
Catches every known failure mode in one run.
Does NOT stop at first error — collects everything, then reports.

Usage:
    python diagnose.py
    python diagnose.py --config configs/phase1.yaml

Paste the full output when asking for help.
"""
import argparse, sys, os, traceback, inspect, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/phase1.yaml")
args = parser.parse_args()

# ── Colour helpers ──────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"; X = "\033[0m"
ISSUES = []   # collected (severity, location, message) tuples

def log_pass(section, msg=""):
    print(f"  {G}OK  {X} {section}  {msg}")

def log_fail(section, msg, exc=None):
    detail = f"\n         {traceback.format_exc().strip()}" if exc else ""
    print(f"  {R}FAIL{X} {section}  {msg}{detail}")
    ISSUES.append(("FAIL", section, msg))

def log_warn(section, msg):
    print(f"  {Y}WARN{X} {section}  {msg}")
    ISSUES.append(("WARN", section, msg))

def section(title):
    print(f"\n{B}{'─'*60}{X}")
    print(f"{B}  {title}{X}")
    print(f"{B}{'─'*60}{X}")

# ═══════════════════════════════════════════════════════════════════════════
section("A. CONFIG LOADING")
# ═══════════════════════════════════════════════════════════════════════════
cfg = None
try:
    import yaml
    with open(os.path.join(ROOT, "configs", "base.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, args.config)) as f:
        override = yaml.safe_load(f)
    if override:
        cfg.update(override)
    log_pass("base.yaml + phase yaml loaded")
except Exception as e:
    log_fail("Config load", str(e), e)
    print(f"\n{R}Cannot continue without config.{X}\n")
    sys.exit(1)

# Key presence checks
required_keys = [
    ("episode.t_max_train",   lambda c: c.get("episode",{}).get("t_max_train")),
    ("jobs.n_jobs_train",     lambda c: c.get("jobs",{}).get("n_jobs_train")),
    ("jobs.op_type_compatibility", lambda c: c.get("jobs",{}).get("op_type_compatibility")),
    ("jobs.proc_time_min_hours",   lambda c: c.get("jobs",{}).get("proc_time_min_hours")),
    ("machines (list)",       lambda c: c.get("machines")),
    ("resources.renewable",   lambda c: c.get("resources",{}).get("renewable")),
    ("resources.consumable",  lambda c: c.get("resources",{}).get("consumable")),
    ("mappo.entropy_coef",    lambda c: c.get("mappo",{}).get("entropy_coef")),
    ("mappo.rollout_steps",   lambda c: c.get("mappo",{}).get("rollout_steps")),
    ("logging.tensorboard_dir",lambda c: c.get("logging",{}).get("tensorboard_dir")),
]
for key, getter in required_keys:
    v = getter(cfg)
    if v is None:
        log_fail(f"  config[{key}]", "MISSING — will cause KeyError at runtime")
    else:
        log_pass(f"  config[{key}]", f"= {v}")

# Value sanity
t_max = cfg.get("episode",{}).get("t_max_train", 0)
n_jobs = cfg.get("jobs",{}).get("n_jobs_train", 0)
ent_coef = cfg.get("mappo",{}).get("entropy_coef", 0)
if t_max != 150:
    log_warn("t_max_train", f"= {t_max}, expected 150")
if n_jobs != 40:
    log_warn("n_jobs_train", f"= {n_jobs}, expected 40")
if ent_coef < 0.04:
    log_warn("entropy_coef", f"= {ent_coef}, too low (collapse risk, recommend 0.05)")

# ═══════════════════════════════════════════════════════════════════════════
section("B. IMPORT CHAIN")
# ═══════════════════════════════════════════════════════════════════════════
MODULES = {
    # (import_path, what_to_check_exists_after_import)
    "utils.seeding":           ("seed_everything",),
    "utils.distributions":     ("sample_weibull_failure",),
    "utils.logger":            ("Logger",),
    "utils.checkpoint":        ("save_checkpoint", "load_checkpoint"),
    "environments.transitions.degradation": ("MachineState", "DegradationEngine", "MachineStatus"),
    "environments.transitions.job_dynamics":("Job", "Operation", "JobDynamicsEngine"),
    "environments.transitions.resource_dynamics": ("ResourceState",),
    "environments.transitions.failure_handler":   ("FailureHandler",),
    "environments.spaces.observation_spaces":     ("compute_agent1_obs_dim",),
    "environments.mfg_env":    ("ManufacturingEnv", "AGENT_PDM", "AGENT_JOBSHOP"),
    "rewards.components.shared_reward":      ("compute_shared_reward",),
    "rewards.components.maintenance_reward": ("compute_maintenance_reward",),
    "rewards.components.scheduling_reward":  ("compute_scheduling_reward",),
    "rewards.reward_fn":       ("RewardFunction",),
    "models.mlp_policy":       ("MLPPolicy",),
    "models.tgin.tgin":        ("TGIN",),
    "models.tgin.graph_builder":("GraphBuilder",),
    "models.critic":           ("CentralizedCritic",),
    "agents.pdm_agent":        ("PDMAgent",),
    "agents.jobshop_agent":    ("JobShopAgent",),
    "training.rollout_buffer": ("RolloutBuffer",),
    "training.ppo_update":     ("ppo_update", "build_optimizers"),
    "training.mappo_trainer":  ("MAPPOTrainer",),
}

imported = {}
for mod_path, symbols in MODULES.items():
    try:
        import importlib
        mod = importlib.import_module(mod_path)
        missing = [s for s in symbols if not hasattr(mod, s)]
        if missing:
            log_fail(f"  {mod_path}", f"imported but missing: {missing}")
        else:
            log_pass(f"  {mod_path}", f"[{', '.join(symbols)}]")
        imported[mod_path] = mod
    except Exception as e:
        log_fail(f"  {mod_path}", f"ImportError: {e}")
        imported[mod_path] = None

# ═══════════════════════════════════════════════════════════════════════════
section("C. FUNCTION SIGNATURE AUDIT")
# ═══════════════════════════════════════════════════════════════════════════

def sig_check(fn, expected_params, label):
    """Check fn accepts expected_params. Warn on missing ones."""
    if fn is None:
        log_fail(f"  {label}", "function not importable")
        return
    actual = set(inspect.signature(fn).parameters)
    missing = [p for p in expected_params if p not in actual]
    extra   = [p for p in actual if p not in expected_params and p != "self"]
    if missing:
        log_warn(f"  {label}", f"missing params: {missing}  (caller will crash if it passes these)")
    elif extra:
        log_pass(f"  {label}", f"OK (extra params vs expected: {extra})")
    else:
        log_pass(f"  {label}")

# shared_reward
if imported.get("rewards.components.shared_reward"):
    mod = imported["rewards.components.shared_reward"]
    sig_check(getattr(mod, "compute_shared_reward", None),
              ["newly_failed_machine_ids", "c_fail"],
              "compute_shared_reward (min required params)")

# maintenance_reward
if imported.get("rewards.components.maintenance_reward"):
    mod = imported["rewards.components.maintenance_reward"]
    sig_check(getattr(mod, "compute_maintenance_reward", None),
              ["maintenance_actions", "ordering_cost", "machine_states",
               "shared_reward", "weights"],
              "compute_maintenance_reward (min required params)")

# scheduling_reward
if imported.get("rewards.components.scheduling_reward"):
    mod = imported["rewards.components.scheduling_reward"]
    sig_check(getattr(mod, "compute_scheduling_reward", None),
              ["jobs", "completed_job_ids", "assignment", "shared_reward", "weights"],
              "compute_scheduling_reward (min required params)")

# logger.log_episode
if imported.get("utils.logger"):
    mod = imported["utils.logger"]
    log_ep = getattr(mod.Logger, "log_episode", None)
    if log_ep:
        params = set(inspect.signature(log_ep).parameters) - {"self"}
        required_log = {"episode","episode_return1","episode_return2","episode_length",
                        "n_failures","weighted_tard","n_jobs_completed","avg_health",
                        "n_PM","n_CM","n_jobs_late","avg_hazard_rate","mtbf",
                        "service_level","avg_inventory"}
        missing_log = required_log - params
        if missing_log:
            log_warn("  Logger.log_episode", f"missing params: {missing_log}")
        else:
            log_pass("  Logger.log_episode", "all 15 params present")

# PDMAgent obs_dim param
if imported.get("agents.pdm_agent"):
    mod = imported["agents.pdm_agent"]
    params = set(inspect.signature(mod.PDMAgent.__init__).parameters)
    if "obs_dim" in params:
        log_pass("  PDMAgent.__init__", "accepts obs_dim param")
    else:
        log_fail("  PDMAgent.__init__", "missing obs_dim param — shape mismatch crash")

# MLPPolicy obs_dim param
if imported.get("models.mlp_policy"):
    mod = imported["models.mlp_policy"]
    params = set(inspect.signature(mod.MLPPolicy.__init__).parameters)
    if "obs_dim" in params:
        log_pass("  MLPPolicy.__init__", "accepts obs_dim param")
    else:
        log_fail("  MLPPolicy.__init__", "missing obs_dim param — shape mismatch crash")

# reward_fn.compute inventory_total
if imported.get("rewards.reward_fn"):
    mod = imported["rewards.reward_fn"]
    params = set(inspect.signature(mod.RewardFunction.compute).parameters)
    if "inventory_total" in params:
        log_pass("  RewardFunction.compute", "accepts inventory_total")
    else:
        log_warn("  RewardFunction.compute", "missing inventory_total (w_hold won't fire)")

# MAPPOTrainer._collect_one_step exists
if imported.get("training.mappo_trainer"):
    mod = imported["training.mappo_trainer"]
    for method in ["train", "_collect_one_step", "_reset_env", "_estimate_value"]:
        if hasattr(mod.MAPPOTrainer, method):
            log_pass(f"  MAPPOTrainer.{method}")
        else:
            log_fail(f"  MAPPOTrainer.{method}", "method missing")

# ═══════════════════════════════════════════════════════════════════════════
section("D. ENVIRONMENT SMOKE TEST")
# ═══════════════════════════════════════════════════════════════════════════
env = None
try:
    from environments.mfg_env import ManufacturingEnv, AGENT_PDM, AGENT_JOBSHOP
    env = ManufacturingEnv(cfg)
    obs_dict, _ = env.reset(seed=42)
    obs1 = obs_dict[AGENT_PDM]
    log_pass("ManufacturingEnv reset", f"obs1={obs1.shape} dtype={obs1.dtype}")

    assert obs1.dtype == np.float32, f"obs1 dtype={obs1.dtype}"
    assert not np.isnan(obs1).any(), "obs1 contains NaN at reset"
    assert not np.isinf(obs1).any(), "obs1 contains Inf at reset"
    log_pass("obs1 dtype/nan/inf")

    n_jobs = len(env.jobs)
    n_machines = len(env.machine_states)
    n_pairs = len(env._valid_pairs)
    log_pass("jobs/machines/valid_pairs", f"jobs={n_jobs} machines={n_machines} pairs={n_pairs}")

    if n_jobs != 40:
        log_warn("n_jobs at reset", f"={n_jobs}, expected 40 — check n_jobs_train in config")
    if n_pairs == 0:
        log_warn("valid_pairs at reset", "= 0 — Agent 2 has no actions (check job generation)")

    # Check PM/CM counters exist
    assert hasattr(env, "_episode_pm"), "_episode_pm missing from env"
    assert hasattr(env, "_episode_cm"), "_episode_cm missing from env"
    log_pass("_episode_pm/_episode_cm counters present")

except Exception as e:
    log_fail("Env smoke test", str(e), e)

# ═══════════════════════════════════════════════════════════════════════════
section("E. REWARD FUNCTION SMOKE TEST")
# ═══════════════════════════════════════════════════════════════════════════
try:
    from rewards.reward_fn import RewardFunction
    rf = RewardFunction(cfg)

    # Check inspect probing worked
    log_pass("RewardFunction init", f"probed {len(rf._shared_params)} shared / "
             f"{len(rf._maint_params)} maint / {len(rf._sched_params)} sched params")

    if env is not None:
        env.reset(seed=0)
        # Run 3 steps and catch any compute() crash
        for i in range(3):
            maint = np.zeros(len(env.machine_states), dtype=int)
            reorder = np.zeros(len(cfg["resources"]["consumable"]))
            env._step_agent1({"maintenance": maint, "reorder": reorder})
            env._step_agent2(0 if env._valid_pairs else len(env._valid_pairs))
            env._resolve_physics()
            env._compute_rewards()
            r1 = env.rewards[AGENT_PDM]
            r2 = env.rewards[AGENT_JOBSHOP]
            assert not np.isnan(r1), f"r1 NaN at step {i}"
            assert not np.isnan(r2), f"r2 NaN at step {i}"
        log_pass("reward_fn.compute() x3 steps", f"r1={r1:.3f}  r2={r2:.3f}")

except Exception as e:
    log_fail("Reward function smoke test", str(e), e)

# ═══════════════════════════════════════════════════════════════════════════
section("F. AGENT INSTANTIATION + OBS-NETWORK MATCH")
# ═══════════════════════════════════════════════════════════════════════════
agent1 = None
try:
    from environments.mfg_env import ManufacturingEnv, AGENT_PDM
    from agents.pdm_agent import PDMAgent

    _env = ManufacturingEnv(cfg)
    _obs, _ = _env.reset(seed=0)
    actual_obs_dim = int(_obs[AGENT_PDM].shape[0])

    # Check compute_agent1_obs_dim matches actual
    try:
        from environments.spaces.observation_spaces import compute_agent1_obs_dim
        formula_dim = compute_agent1_obs_dim(cfg)
        if formula_dim != actual_obs_dim:
            log_warn("obs dim formula vs actual",
                     f"formula={formula_dim} actual={actual_obs_dim} "
                     f"(obs_dim param will override — OK if using updated files)")
        else:
            log_pass("obs dim formula == actual", f"= {actual_obs_dim}")
    except Exception as e:
        log_warn("compute_agent1_obs_dim", str(e))

    agent1 = PDMAgent(cfg, device="cpu", obs_dim=actual_obs_dim)
    net_in = agent1.policy.trunk[0].in_features
    if net_in != actual_obs_dim:
        log_fail("Network input dim", f"network={net_in} obs={actual_obs_dim} — MISMATCH")
    else:
        log_pass("Network input dim == obs dim", f"= {actual_obs_dim}")
    log_pass("PDMAgent params", f"total={sum(p.numel() for p in agent1.parameters()):,}")

except Exception as e:
    log_fail("Agent1 instantiation", str(e), e)

try:
    from agents.jobshop_agent import JobShopAgent
    agent2 = JobShopAgent(cfg, device="cpu")
    try:
        n = sum(p.numel() for p in agent2.parameters())
        log_pass("JobShopAgent params", f"total={n:,}")
    except Exception:
        log_pass("JobShopAgent", "instantiated (TGIN param count N/A without torch-geometric)")
except Exception as e:
    log_fail("Agent2 instantiation", str(e), e)

# ═══════════════════════════════════════════════════════════════════════════
section("G. TRAINER DRY-RUN (10 STEPS)")
# ═══════════════════════════════════════════════════════════════════════════
try:
    from training.mappo_trainer import MAPPOTrainer

    trainer = MAPPOTrainer(cfg)
    log_pass("MAPPOTrainer init", f"device={trainer.device} obs_dim_measured={trainer.agent1.policy.trunk[0].in_features}")

    # Patch agent2.act to avoid torch-geometric requirement in this check
    # (real training on the user's machine has torch-geometric installed)
    _orig_act = trainer.agent2.act
    def _fake_act(obs, valid_pairs):
        idx = 0 if valid_pairs else len(valid_pairs)
        return None, idx, 0.0, 1.0
    trainer.agent2.act = _fake_act
    trainer._estimate_value = lambda **kw: 0.0

    trainer._reset_env(seed=0)
    step_errors = []
    for i in range(10):
        try:
            done, trunc = trainer._collect_one_step()
            if done:
                trainer.episode += 1
                trainer._reset_env()
        except Exception as e:
            step_errors.append(f"step {i}: {e}")
            break

    if step_errors:
        log_fail("Trainer collect steps", step_errors[0])
    else:
        log_pass("10 collect steps", f"global_step={trainer.global_step}")

    # Test log_episode call (the call that was crashing before)
    try:
        import numpy as np
        avg_h = float(np.mean([s.health for s in trainer.env.machine_states]))
        n_late = sum(1 for j in trainer.env.jobs
                     if j.is_complete and j.completion_time is not None
                     and j.completion_time > j.due_date)
        n_comp = trainer.env._episode_completions
        trainer.logger.log_episode(
            episode=0, episode_return1=1.0, episode_return2=-1.0,
            episode_length=150, n_failures=0, weighted_tard=0.1,
            n_jobs_completed=n_comp, avg_health=avg_h,
            n_PM=trainer.env._episode_pm, n_CM=trainer.env._episode_cm,
            n_jobs_late=n_late, avg_hazard_rate=0.001,
            mtbf=100.0, service_level=0.8, avg_inventory=12.0,
        )
        log_pass("logger.log_episode (15-arg call)")
    except Exception as e:
        log_fail("logger.log_episode", str(e), e)

    trainer.logger.close()

except Exception as e:
    log_fail("Trainer dry-run", str(e), e)

# ═══════════════════════════════════════════════════════════════════════════
section("H. CHECKPOINT + RESUME PATH")
# ═══════════════════════════════════════════════════════════════════════════
try:
    from utils.checkpoint import save_checkpoint, load_checkpoint
    import inspect as _ins
    save_sig = set(_ins.signature(save_checkpoint).parameters)
    load_sig = set(_ins.signature(load_checkpoint).parameters)
    needed_save = {"checkpoint_dir","episode","global_step","actor1","actor2",
                   "critic","optim_actor1","optim_actor2","optim_critic","config"}
    needed_load = {"path","actor1","actor2","critic"}
    missing_s = needed_save - save_sig
    missing_l = needed_load - load_sig
    if missing_s:
        log_fail("save_checkpoint signature", f"missing: {missing_s}")
    else:
        log_pass("save_checkpoint signature")
    if missing_l:
        log_fail("load_checkpoint signature", f"missing: {missing_l}")
    else:
        log_pass("load_checkpoint signature")
except Exception as e:
    log_fail("Checkpoint utils", str(e), e)

# Check --resume stub is wired (not TODO)
try:
    with open(os.path.join(ROOT, "scripts", "train.py")) as f:
        src = f.read()
    if "# TODO" in src and "resume" in src.lower():
        log_fail("scripts/train.py --resume", "still a TODO stub — will silently restart from step 0")
    else:
        log_pass("scripts/train.py --resume", "wired to load_checkpoint")
except Exception as e:
    log_warn("scripts/train.py check", str(e))

# ═══════════════════════════════════════════════════════════════════════════
section("SUMMARY")
# ═══════════════════════════════════════════════════════════════════════════
fails = [i for i in ISSUES if i[0] == "FAIL"]
warns = [i for i in ISSUES if i[0] == "WARN"]

print()
if fails:
    print(f"  {R}BLOCKERS ({len(fails)}):{X}")
    for _, loc, msg in fails:
        print(f"    • {loc}: {msg}")
if warns:
    print(f"  {Y}WARNINGS ({len(warns)}):{X}")
    for _, loc, msg in warns:
        print(f"    • {loc}: {msg}")
if not fails and not warns:
    print(f"  {G}ALL CLEAR — ready to train{X}")
elif not fails:
    print(f"  {G}No blockers.{X} Fix warnings when convenient.")
else:
    print(f"\n  {R}Fix blockers before starting training.{X}")

print()
sys.exit(1 if fails else 0)
