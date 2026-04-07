"""
test_resources_exhaustive.py
============================
Exhaustive resource dynamics test over 3000 environment steps.
Verifies every resource flow: depletion, ordering, delivery,
renewable hold/free, CM blocking, reward signals.

Usage: python test_resources_exhaustive.py
Pass:  All assertions green, summary table printed
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml, numpy as np

with open("configs/base.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

from environments.mfg_env import ManufacturingEnv
from environments.transitions.degradation import MachineStatus
from environments.spaces.action_spaces import (
    build_agent1_maintenance_mask, build_agent1_reorder_mask, ACTION_PM)

env = ManufacturingEnv(cfg)

TESTS = []  # (name, passed, detail)

def check(name, cond, detail=""):
    TESTS.append((name, cond, detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}")
    if detail and not cond:
        print(f"         {detail}")

print("\n" + "="*65)
print("  EXHAUSTIVE RESOURCE DYNAMICS TEST — 3000 steps")
print("="*65)

# ─────────────────────────────────────────────────────────────────
# PART A: Depletion correctness (single episode, forced events)
# ─────────────────────────────────────────────────────────────────
print("\n--- PART A: Depletion & Consumption ---")

env.reset(seed=42)
inv0    = env.resource_state.consumable_inventory.copy()
ren0    = env.resource_state.renewable_available.copy()
K_ren   = env.resource_state.renewable_capacity.copy()
rho_pm  = env.rho_PM[0, env.n_renewable:]   # consumable part
rho_cm  = env.rho_CM[0, env.n_renewable:]
rho_pm_ren = env.rho_PM[0, :env.n_renewable]
rho_cm_ren = env.rho_CM[0, :env.n_renewable]

# A1: PM consumes correct consumable amount
maint = np.zeros(5, int); maint[0] = 1
env._step_agent1({"maintenance": maint, "reorder": np.zeros(3)})
env._step_agent2(0 if env._valid_pairs else 0)
env._resolve_physics(); env._compute_rewards()
inv1 = env.resource_state.consumable_inventory
check("A1: PM deducts exactly rho_PM consumable",
      np.allclose(inv0 - inv1, rho_pm, atol=0.01),
      f"expected delta={rho_pm}, got {(inv0-inv1).round(2)}")

# A2: PM reduces renewable (via env recompute, not _consume)
ren1 = env.resource_state.renewable_available
check("A2: PM reduces renewable by rho_PM_ren",
      np.allclose(ren0 - ren1, rho_pm_ren, atol=0.1),
      f"expected delta={rho_pm_ren}, got {(ren0-ren1)}")

# A3: Renewable freed when PM completes
tau_pm = cfg["machines"][0]["tau_PM_shifts"]
for _ in range(tau_pm + 1):
    env._step_agent1({"maintenance": np.zeros(5,int), "reorder": np.zeros(3)})
    env._step_agent2(0 if env._valid_pairs else 0)
    env._resolve_physics(); env._compute_rewards()
ren_after_pm = env.resource_state.renewable_available
check("A3: Renewable freed after PM completes",
      np.allclose(ren_after_pm, K_ren, atol=0.5),
      f"expected {K_ren}, got {ren_after_pm} after PM+{tau_pm}steps")

# A4: Auto-CM deducts consumable (2x PM)
env.reset(seed=42)
inv0 = env.resource_state.consumable_inventory.copy()
env.machine_states[1].status = MachineStatus.FAIL
env.machine_states[1].health = 5.0
env._cm_queue = {1}
n_cm = env._attempt_auto_cm()
inv_after_cm = env.resource_state.consumable_inventory
check("A4: auto-CM deducts rho_CM consumable",
      np.allclose(inv0 - inv_after_cm, rho_cm, atol=0.01),
      f"expected delta={rho_cm}, got {(inv0-inv_after_cm).round(2)}")

# A5: Auto-CM does NOT double-deduct renewable
ren_after_cm = env.resource_state.renewable_available
exp_ren = K_ren - rho_cm_ren
check("A5: auto-CM deducts renewable exactly once",
      np.allclose(ren_after_cm, exp_ren, atol=0.1),
      f"expected {exp_ren}, got {ren_after_cm}")

# A6: Renewable freed after CM completes
tau_cm = cfg["machines"][1]["tau_CM_shifts"]
env.reset(seed=42)
env.machine_states[1].status = MachineStatus.FAIL
env._cm_queue = {1}
env._attempt_auto_cm()
for _ in range(tau_cm + 2):
    env._step_agent1({"maintenance": np.zeros(5,int), "reorder": np.zeros(3)})
    env._step_agent2(0 if env._valid_pairs else 0)
    env._resolve_physics(); env._compute_rewards()
ren_after = env.resource_state.renewable_available
check("A6: Renewable freed after CM completes",
      np.allclose(ren_after, K_ren, atol=0.5),
      f"expected {K_ren}, got {ren_after} after tau_CM={tau_cm}")

# A7: CM blocked when no resources
env.reset(seed=42)
env.resource_state.renewable_available[:] = 0
env.machine_states[0].status = MachineStatus.FAIL
env._cm_queue = {0}
n_started = env._attempt_auto_cm()
check("A7: auto-CM blocked when renewables=0",
      n_started == 0 and 0 in env._cm_queue,
      f"started={n_started} queue={env._cm_queue}")

# A8: CM retries next step when resources free
env.resource_state.renewable_available = K_ren.copy()
n_retry = env._attempt_auto_cm()
check("A8: auto-CM retries and succeeds when resources freed",
      n_retry == 1 and env.machine_states[0].status == MachineStatus.CM,
      f"started={n_retry}")

# ─────────────────────────────────────────────────────────────────
# PART B: Ordering pipeline
# ─────────────────────────────────────────────────────────────────
print("\n--- PART B: Ordering Pipeline ---")

env.reset(seed=42)
L = cfg["resources"]["consumable"][0]["lead_time_shifts"]  # =5

# B1: Reorder mask allows ordering at start
mask = build_agent1_reorder_mask(env.resource_state, env.rho_CM_max)
check("B1: Reorder mask allows at least 1 resource at start",
      mask.any(), f"mask={mask}")

# B2: Order placed lands in pipeline at correct lag
inv_pre = env.resource_state.consumable_inventory.copy()
env._step_agent1({"maintenance": np.zeros(5,int), "reorder": np.array([10., 0., 0.])})
env._step_agent2(0 if env._valid_pairs else 0)
env._resolve_physics(); env._compute_rewards()
pipeline_sum = env.resource_state.pending_orders[0].sum()
check("B2: Order quantity placed in pipeline",
      abs(pipeline_sum - 10.0) < 0.01,
      f"pipeline_sum={pipeline_sum:.1f} expected 10")

# B3: Order arrives after exactly lead_time steps
for step in range(L - 1):
    env._step_agent1({"maintenance": np.zeros(5,int), "reorder": np.zeros(3)})
    env._step_agent2(0 if env._valid_pairs else 0)
    env._resolve_physics(); env._compute_rewards()
inv_before = env.resource_state.consumable_inventory.copy()
# One more step — should arrive now
env._step_agent1({"maintenance": np.zeros(5,int), "reorder": np.zeros(3)})
env._step_agent2(0 if env._valid_pairs else 0)
env._resolve_physics(); env._compute_rewards()
inv_after = env.resource_state.consumable_inventory
arrived = inv_after[0] - inv_before[0]
check("B3: Order arrives after lead_time steps",
      arrived >= 9.5,
      f"arrived={arrived:.1f} expected ~10 after L={L} steps")

# B4: Ordering cost is positive and proportional to qty
env.reset(seed=42)
env._step_agent1({"maintenance": np.zeros(5,int), "reorder": np.array([5., 3., 0.])})
env._step_agent2(0); env._resolve_physics(); env._compute_rewards()
oc = env._last_ordering_cost
exp_oc = 5*cfg["resources"]["consumable"][0]["reorder_cost"] + \
         3*cfg["resources"]["consumable"][1]["reorder_cost"]
check("B4: Ordering cost = qty * reorder_cost",
      abs(oc - exp_oc) < 0.01,
      f"got {oc:.2f} expected {exp_oc:.2f}")

# B5: Consumable NEVER goes negative
env.reset(seed=1)
min_inv = float('inf')
for _ in range(150):
    maint = np.zeros(5,int)
    for i, s in enumerate(env.machine_states):
        if s.status == MachineStatus.OP and not env.machine_busy[i] and s.health < 30:
            maint[i] = 1
    reorder = np.where(build_agent1_reorder_mask(env.resource_state, env.rho_CM_max), 8., 0.)
    env._step_agent1({"maintenance": maint, "reorder": reorder})
    env._step_agent2(0 if env._valid_pairs else 0)
    env._resolve_physics(); env._compute_rewards()
    min_inv = min(min_inv, env.resource_state.consumable_inventory.min())
check("B5: Consumable never goes negative (150 steps)",
      min_inv >= -0.01,
      f"min_inventory={min_inv:.3f}")

# B6: Renewable never goes negative
env.reset(seed=2)
min_ren = float('inf')
for _ in range(150):
    maint = np.zeros(5,int)
    for i, s in enumerate(env.machine_states):
        if s.status == MachineStatus.OP and not env.machine_busy[i] and s.health < 40:
            maint[i] = 1
    env._step_agent1({"maintenance": maint, "reorder": np.zeros(3)})
    env._step_agent2(0 if env._valid_pairs else 0)
    env._resolve_physics(); env._compute_rewards()
    min_ren = min(min_ren, env.resource_state.renewable_available.min())
check("B6: Renewable never goes negative (150 steps)",
      min_ren >= 0,
      f"min_renewable={min_ren}")

# ─────────────────────────────────────────────────────────────────
# PART C: Reward signals tied to resource events
# ─────────────────────────────────────────────────────────────────
print("\n--- PART C: Reward Signals ---")

import yaml as _y
with open("rewards/reward_weights.yaml") as f:
    w = _y.safe_load(f)

c_CM   = w["c_CM"];  c_PM   = w["c_PM"]
c_fail = w["c_fail"]; lam   = w["lambda_shared"]
w_hold = w["w_hold"]

# C1: PM costs c_PM in r1
env.reset(seed=42)
env._step_agent1({"maintenance": np.zeros(5,int), "reorder": np.zeros(3)})
env._step_agent2(0); env._resolve_physics(); env._compute_rewards()
r1_noop = list(env.rewards.values())[0]

env.reset(seed=42)
maint = np.zeros(5,int); maint[0] = 1
env._step_agent1({"maintenance": maint, "reorder": np.zeros(3)})
env._step_agent2(0); env._resolve_physics(); env._compute_rewards()
r1_pm = list(env.rewards.values())[0]
check("C1: PM action reduces r1 by c_PM",
      (r1_noop - r1_pm) >= c_PM * 0.75,  # w_hazard changes hazard component too
      f"r1_noop={r1_noop:.3f} r1_pm={r1_pm:.3f} diff={r1_noop-r1_pm:.3f} expected>={c_PM*0.75:.2f}")

# C2: auto-CM charges c_CM to r1
env.reset(seed=42)
inv0 = env.resource_state.consumable_inventory.copy()

env._step_agent1({"maintenance": np.zeros(5,int), "reorder": np.zeros(3)})
env._step_agent2(0); env._resolve_physics(); env._compute_rewards()
r1_baseline = list(env.rewards.values())[0]

env.reset(seed=42)
env.machine_states[0].status = MachineStatus.FAIL
env.machine_states[0].health = 5.0
env._cm_queue = {0}
env._step_agent1({"maintenance": np.zeros(5,int), "reorder": np.zeros(3)})
env._step_agent2(0); env._resolve_physics(); env._compute_rewards()
r1_with_cm = list(env.rewards.values())[0]
auto_cm_count = env._auto_cm_count
check("C2: auto-CM deducts c_CM from r1",
      auto_cm_count > 0 and (r1_baseline - r1_with_cm) >= c_CM * 0.5,
      f"auto_cm={auto_cm_count} r1_base={r1_baseline:.3f} r1_cm={r1_with_cm:.3f}")

# C3: r_shared fires on machine failure
# Force a failure by directly injecting it into _newly_failed before reward compute
env.reset(seed=42)
env._newly_failed = [2]  # machine 2 just failed this step
env.machine_states[2].status = MachineStatus.FAIL
env._compute_rewards()
check("C3: r_shared < 0 when machine fails",
      env._last_r_shared < 0,
      f"r_shared={env._last_r_shared:.3f} (should be negative on failure)")

# C4: Ordering cost deducted from r1
env.reset(seed=42)
env._step_agent1({"maintenance": np.zeros(5,int), "reorder": np.zeros(3)})
env._step_agent2(0); env._resolve_physics(); env._compute_rewards()
r1_no_order = list(env.rewards.values())[0]

env.reset(seed=42)
env._step_agent1({"maintenance": np.zeros(5,int), "reorder": np.array([5.,0.,0.])})
env._step_agent2(0); env._resolve_physics(); env._compute_rewards()
r1_with_order = list(env.rewards.values())[0]
reorder_cost_per5 = 5 * cfg["resources"]["consumable"][0]["reorder_cost"]  # 5*10=50
delta_c    = w.get("delta_obj", 0.5)
exp_penalty = delta_c * reorder_cost_per5
check("C4: Ordering cost deducted from r1 (delta-weighted)",
      (r1_no_order - r1_with_order) >= exp_penalty * 0.9,
      f"diff={r1_no_order-r1_with_order:.2f} expected>={exp_penalty:.2f}")

# C5: Holding cost proportional to inventory
env.reset(seed=42)
inv_high = env.resource_state.consumable_inventory.copy()
total_inv_high = inv_high.sum()
env.resource_state.consumable_inventory = np.zeros(3)
env._step_agent1({"maintenance": np.zeros(5,int), "reorder": np.zeros(3)})
env._step_agent2(0); env._resolve_physics(); env._compute_rewards()
r1_low_inv = list(env.rewards.values())[0]

env.reset(seed=42)
env._step_agent1({"maintenance": np.zeros(5,int), "reorder": np.zeros(3)})
env._step_agent2(0); env._resolve_physics(); env._compute_rewards()
r1_high_inv = list(env.rewards.values())[0]
# C5 intent changed: stockout penalty now makes low inventory WORSE (correct behaviour)
# Verify that stockout penalty fires correctly (inv=0 costs far more than normal)
check("C5: Stockout penalty fires (zero inv penalised more than full inv)",
      r1_high_inv > r1_low_inv,  # normal inv better than zero (stockout penalised)
      f"r1_normal={r1_high_inv:.3f} r1_zero={r1_low_inv:.3f} — stockout should penalise")

# ─────────────────────────────────────────────────────────────────
# PART D: Long-run stability (3000 steps)
# ─────────────────────────────────────────────────────────────────
print("\n--- PART D: Long-run Stability (3000 steps) ---")

env.reset(seed=99)
stats = {"cm": 0, "pm": 0, "fail": 0, "orders": 0, "order_cost": 0.0,
         "renewable_neg": 0, "consumable_neg": 0, "r_shared_nonzero": 0}
r1_history = []; r2_history = []; r_shared_history = []
inv_history = []
ren_history  = []

for step in range(3000):
    # Episode reset every 150 steps
    if step > 0 and step % 150 == 0:
        env.reset(seed=step)

    # Policy: PM when health < 50, order when mask allows
    maint = np.zeros(5, int)
    for i, s in enumerate(env.machine_states):
        if s.status == MachineStatus.OP and not env.machine_busy[i] and s.health < 50:
            if build_agent1_maintenance_mask(env.machine_states, env.machine_busy,
                                              env.resource_state, env.rho_PM,
                                              env.rho_CM, env.n_renewable)[i, 1]:
                maint[i] = 1

    reorder_mask = build_agent1_reorder_mask(env.resource_state, env.rho_CM_max)
    reorder = np.where(reorder_mask, 8.0, 0.0)
    if reorder.any(): stats["orders"] += 1

    env._step_agent1({"maintenance": maint, "reorder": reorder})
    env._step_agent2(0 if env._valid_pairs else 0)
    env._resolve_physics()
    env._compute_rewards()

    rewards = list(env.rewards.values())
    r1 = rewards[0] if len(rewards) > 0 else 0.0
    r2 = rewards[1] if len(rewards) > 1 else 0.0
    rs = getattr(env, "_last_r_shared", 0.0)

    r1_history.append(r1); r2_history.append(r2); r_shared_history.append(rs)

    stats["cm"]    += getattr(env, "_auto_cm_count", 0)
    stats["pm"]    += int(maint.sum())
    stats["fail"]  += len(getattr(env, "_newly_failed", []))
    stats["order_cost"] += getattr(env, "_last_ordering_cost", 0.0)

    if rs != 0: stats["r_shared_nonzero"] += 1
    if env.resource_state.renewable_available.min() < 0:
        stats["renewable_neg"] += 1
    if env.resource_state.consumable_inventory.min() < -0.01:
        stats["consumable_neg"] += 1

    inv_history.append(env.resource_state.consumable_inventory.sum())
    ren_history.append(env.resource_state.renewable_available.sum())

r1_arr = np.array(r1_history); r2_arr = np.array(r2_history)
rs_arr = np.array(r_shared_history)

# D1: CM happened during 3000 steps
check("D1: Auto-CM events > 0 over 3000 steps",
      stats["cm"] > 0, f"cm={stats['cm']}")

# D2: Orders happened
check("D2: Ordering happens (order_cost > 0)",
      stats["order_cost"] > 0,
      f"order_cost={stats['order_cost']:.1f}")

# D3: r_shared fires on failures
if stats["fail"] > 0:
    check("D3: r_shared non-zero when failures occurred",
          stats["r_shared_nonzero"] > 0,
          f"failures={stats['fail']} r_shared_nonzero={stats['r_shared_nonzero']}")

# D4: Renewables never negative
check("D4: Renewable never negative (3000 steps)",
      stats["renewable_neg"] == 0,
      f"negative steps: {stats['renewable_neg']}")

# D5: Consumables never negative
check("D5: Consumable never negative (3000 steps)",
      stats["consumable_neg"] == 0,
      f"negative steps: {stats['consumable_neg']}")

# D6: r1 not exploding
check("D6: r1 not exploding (|mean| < 100)",
      abs(r1_arr.mean()) < 100,
      f"r1 mean={r1_arr.mean():.2f}")

# D7: PM/CM ratio reasonable
pm_cm_ratio = stats["pm"] / max(stats["cm"], 1)
check("D7: PM/CM ratio > 0 (proactive maintenance happening)",
      stats["pm"] > 0,
      f"PM={stats['pm']} CM={stats['cm']} ratio={pm_cm_ratio:.1f}")

# D8: Inventory stays positive (not zero-stuck)
inv_mean = np.mean(inv_history)
check("D8: Average inventory > 0 (ordering replenishes correctly)",
      inv_mean > 1.0,
      f"avg total inventory={inv_mean:.1f}")

# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SUMMARY")
print("="*65)
print(f"  Total steps: 3000")
print(f"  Failures:    {stats['fail']}")
print(f"  Auto-CM:     {stats['cm']}")
print(f"  PM events:   {stats['pm']}")
print(f"  Orders placed: {stats['orders']}  total_cost={stats['order_cost']:.1f}")
print(f"  r_shared fired: {stats['r_shared_nonzero']} steps")
print()
print(f"  r1:  mean={r1_arr.mean():.3f}  std={r1_arr.std():.3f}  CV={r1_arr.std()/max(abs(r1_arr.mean()),0.01):.2f}")
print(f"  r2:  mean={r2_arr.mean():.3f}  std={r2_arr.std():.3f}")
print(f"  r_shared: mean={rs_arr.mean():.4f}  nonzero={stats['r_shared_nonzero']}")
print(f"  avg inv:  {np.mean(inv_history):.1f}  avg_ren: {np.mean(ren_history):.1f}")
print()
passed = sum(1 for _, ok, _ in TESTS if ok)
failed = sum(1 for _, ok, _ in TESTS if not ok)
total  = len(TESTS)
print(f"  Tests: {passed}/{total} passed", "" if failed == 0 else f"  ({failed} FAILED)")
print()
if failed == 0:
    print("  ALL RESOURCE DYNAMICS TESTS PASSED ✓")
    print("  Environment is ready for training.")
else:
    print("  SOME TESTS FAILED — fix before training.")
    sys.exit(1)