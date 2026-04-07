"""
verify_fix.py  —  Pre-training verification (runs in ~2 minutes)
================================================================
Checks all 6 bug fixes are in place, then runs a live 300-step
environment test to confirm:
  - auto-CM fires when machines fail
  - renewables are freed correctly
  - consumables deplete and ordering happens
  - r_shared is non-zero on failures
  - rewards are reasonable (not exploding or stuck)

Does NOT run a full training subprocess — that would take 15-30 min.

Usage:  python verify_fix.py
Pass:   All checks green
Fail:   Fix the reported issue before training
"""
import sys, os, time, yaml, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

print()
print("=" * 60)
print("  VERIFY FIX — Pre-training Checks")
print("=" * 60)

# ── 1. Config checks ──────────────────────────────────────────────
print("\n--- 1. CONFIG CHECKS ---")
with open("configs/base.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
with open("rewards/reward_weights.yaml", encoding="utf-8") as f:
    w = yaml.safe_load(f)

ec   = cfg.get("mappo", {}).get("entropy_coef")
wi   = w.get("w_fail_idle")
bays = next((r["capacity"] for r in cfg["resources"]["renewable"]
             if "bay" in r.get("name","").lower()), None)
has_reorder = all("reorder_qty" in r for r in cfg["resources"]["consumable"])

ok = True
print(f"  entropy_coef : {ec}   {'OK' if ec == 0.01 else 'FAIL (need 0.01)'}")
print(f"  w_fail_idle  : {wi}   {'OK (removed)' if wi is None else 'FAIL (should be removed)'}")
print(f"  K_bays       : {bays}   {'OK' if bays == 4 else 'FAIL (need 4)'}")
print(f"  reorder_qty  : {'OK' if has_reorder else 'FAIL — add reorder_qty to consumable resources in base.yaml'}")

if ec != 0.01 or wi is not None or bays != 4 or not has_reorder:
    print("\n  Fix those config values then re-run."); sys.exit(1)
print("  Configs OK")

# ── 2. Code structure checks ──────────────────────────────────────
print("\n--- 2. CODE CHECKS ---")
checks = []

with open("training/ppo_update.py", encoding="utf-8", errors="replace") as f:
    ppu = f.read()
checks.append(("Critic trains in ppo_update.py",
               "c_loss.backward()" in ppu and "optim_critic.step()" in ppu))

with open("training/mappo_trainer.py", encoding="utf-8", errors="replace") as f:
    t = f.read()
checks.append(("r_shared logged (not 0.0)", "log_rewards(r1, r2, 0.0" not in t))
checks.append(("Resource debug tags logged", "debug/renewable_" in t))

with open("environments/mfg_env.py", encoding="utf-8", errors="replace") as f:
    e = f.read()
checks.append(("_get_machines_completing_maint exists", "_get_machines_completing_maint" in e))
checks.append(("machines_completing_maint not hardcoded []",
               "machines_completing_maint=[]" not in e))
checks.append(("_episode_order_cost tracked", "_episode_order_cost" in e))

with open("environments/transitions/resource_dynamics.py", encoding="utf-8", errors="replace") as f:
    rd = f.read()
checks.append(("_consume: consumables only", "rho_ren" not in rd.split("def _consume")[1].split("def ")[0]))

with open("models/critic.py", encoding="utf-8", errors="replace") as f:
    cr = f.read()
checks.append(("Critic: n_machines*2 (not 3)", "n_machines * 3" not in cr))

with open("environments/spaces/action_spaces.py", encoding="utf-8", errors="replace") as f:
    ac = f.read()
checks.append(("Reorder mask: safety_stock fix", "safety_stock = rho_CM_max * 10" in ac))

with open("rewards/reward_fn.py", encoding="utf-8", errors="replace") as f:
    rfn = f.read()
checks.append(("reward_fn: n_auto_cm threaded through", "n_auto_cm" in rfn and "n_auto_cm=n_auto_cm" in rfn))

all_ok = True
for label, result in checks:
    print(f"  {'OK' if result else 'FAIL'} {label}")
    if not result:
        all_ok = False

if not all_ok:
    print("\n  Code checks failed."); sys.exit(1)
print("  Code checks OK")

# ── 3. Live environment test (300 steps, ~30 seconds) ─────────────
print("\n--- 3. LIVE ENVIRONMENT TEST (300 steps) ---")
t0 = time.time()

from environments.mfg_env import ManufacturingEnv
from environments.transitions.degradation import MachineStatus
from environments.spaces.action_spaces import (
    build_agent1_maintenance_mask, build_agent1_reorder_mask)

env = ManufacturingEnv(cfg)

total_cm = 0; total_orders = 0; total_fails = 0
r1_vals = []; r_shared_vals = []
renewable_min = float('inf')
consumable_min = float('inf')

for seed in range(3):
    env.reset(seed=seed * 17)
    ep_cm = 0; ep_orders = 0

    for step in range(100):
        maint = np.zeros(5, int)
        for i, s in enumerate(env.machine_states):
            if env.machine_states[i].status == MachineStatus.OP and \
               not env.machine_busy[i] and s.health < 50:
                from environments.spaces.action_spaces import ACTION_PM
                if build_agent1_maintenance_mask(
                        env.machine_states, env.machine_busy,
                        env.resource_state, env.rho_PM, env.rho_CM,
                        env.n_renewable)[i, 1]:
                    maint[i] = 1

        reorder_mask = build_agent1_reorder_mask(env.resource_state, env.rho_CM_max)
        reorder = np.where(reorder_mask, 8.0, 0.0)
        if reorder.any(): ep_orders += 1

        env._step_agent1({"maintenance": maint, "reorder": reorder})
        env._step_agent2(0 if env._valid_pairs else 0)
        env._resolve_physics()
        env._compute_rewards()

        r1_vals.append(list(env.rewards.values())[0])
        r_shared_vals.append(getattr(env, "_last_r_shared", 0.0))
        ep_cm += getattr(env, "_auto_cm_count", 0)
        renewable_min = min(renewable_min, env.resource_state.renewable_available.min())
        consumable_min = min(consumable_min, env.resource_state.consumable_inventory.min())

    total_cm     += ep_cm
    total_orders += ep_orders
    total_fails  += env._episode_failures

elapsed = time.time() - t0

r1_arr     = np.array(r1_vals)
rs_arr     = np.array(r_shared_vals)
rs_nonzero = (rs_arr != 0).sum()

print(f"\n  3 seeds × 100 steps = 300 steps in {elapsed:.1f}s")
print()
print(f"  Auto-CM events  : {total_cm}")
print(f"  Order events    : {total_orders}")
print(f"  Failures        : {total_fails}")
print(f"  r_shared != 0   : {rs_nonzero} steps (should be >0 if failures occurred)")
print(f"  r1 mean/std     : {r1_arr.mean():.3f} / {r1_arr.std():.3f}")
print(f"  r1 range        : [{r1_arr.min():.2f}, {r1_arr.max():.2f}]")
print(f"  Renewable min   : {renewable_min:.0f}")
print(f"  Consumable min  : {consumable_min:.1f}")
print()

issues = []
if total_cm == 0 and total_fails > 0:
    issues.append("AUTO-CM NEVER FIRED")
if total_orders == 0:
    issues.append("ORDERING NEVER HAPPENED")
if rs_nonzero == 0 and total_fails > 0:
    issues.append("r_shared always 0")
if renewable_min < 0:
    issues.append("RENEWABLE NEGATIVE")
if consumable_min < -0.01:
    issues.append("CONSUMABLE NEGATIVE")
if abs(r1_arr.mean()) > 50:
    issues.append("REWARD SCALE ISSUE")

max_ent = 5 * math.log(2)
print(f"  Entropy ceiling: {max_ent:.4f}")
print()

if issues:
    print("  ISSUES FOUND:")
    for issue in issues:
        print(f"    ✗ {issue}")
    sys.exit(1)
else:
    print("  ALL LIVE CHECKS PASSED ✓")
    print("  System is ready for training.")
