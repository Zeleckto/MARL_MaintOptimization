"""
analytics/episode_kpis.py
=========================
Single source of truth for all KPI computation from one episode.
Used by analyze_baselines, analyze_training, global_analytics.

Conference-grade metrics:
  Reliability Engineering (IEEE Trans. Reliability, JQME)
  Job Shop Scheduling (IJPR, EJOR)
  Inventory/Resource (IISE Transactions, OR)
"""
import numpy as np, math, yaml, os

_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "rewards", "reward_weights.yaml")

def _load_weights():
    with open(_WEIGHTS_PATH) as f:
        return yaml.safe_load(f)

def compute_episode_kpis(env, steps: int) -> dict:
    """
    Compute all KPIs after one episode. Call after env episode ends.
    Returns flat dict — all values are Python float.
    """
    cfg  = env.config
    T    = max(steps, 1)

    # ── Job Shop Scheduling ───────────────────────────────────────
    completed  = [j for j in env.jobs if j.completion_time is not None]
    n_done     = len(completed)
    on_time    = [j for j in completed if j.tardiness == 0]

    tard_vals  = [j.weight * j.tardiness for j in completed]
    flow_times = [j.completion_time - j.release_time for j in completed]
    comp_times = [j.completion_time for j in completed]

    makespan      = float(max(comp_times)) if comp_times else float(T)
    wt_sum        = float(sum(tard_vals))
    service_level = float(len(on_time) / max(n_done, 1))
    mean_flow     = float(np.mean(flow_times)) if flow_times else 0.0
    throughput    = float(n_done / T)  # jobs per step

    # Normalised WT: divide by episode length to make comparable across configs
    wt_norm = wt_sum / max(T, 1)

    # Machine utilisation: cumulative op time / (T * dt_hours)
    dt = cfg.get("episode", {}).get("dt_hours", 8.0)
    util_list = [min(getattr(s, "cumulative_op_time", 0.0) / max(T * dt, 1), 1.0)
                 for s in env.machine_states]
    machine_utilisation = float(np.mean(util_list))

    # ── Reliability Engineering ───────────────────────────────────
    from environments.transitions.degradation import MachineStatus
    n_fail = env._episode_failures
    n_PM   = env._episode_pm
    n_CM   = env._episode_cm

    avail_list = [1.0 if s.status == MachineStatus.OP else 0.0
                  for s in env.machine_states]
    availability = float(np.mean(avail_list))

    # MTBF (shifts): total operational time / number of failures
    total_op = sum(getattr(s, "cumulative_op_time", 0.0) / dt
                   for s in env.machine_states)
    mtbf = float(total_op / max(n_fail, 1))

    # MTTR: mean tau_CM from config
    taus_cm = [m.get("tau_CM_shifts", 6) for m in cfg.get("machines", [])]
    mttr = float(np.mean(taus_cm))

    # Inherent availability A_i = MTBF / (MTBF + MTTR)   [ISO 60300]
    inherent_avail = mtbf / max(mtbf + mttr, 1e-8)

    mean_health   = float(np.mean([s.health for s in env.machine_states]))
    mean_rul_norm = float(np.mean([s.rul / max(s.eta, 1) for s in env.machine_states]))
    mean_hazard   = float(np.mean([getattr(s, "hazard_rate", 0.0) for s in env.machine_states]))
    pm_cm_ratio   = float(n_PM / max(n_CM, 1))

    # Fleet health coefficient of variation (spread of degradation)
    healths = [s.health for s in env.machine_states]
    health_cv = float(np.std(healths) / max(np.mean(healths), 1e-8))

    # ── Cost / Inventory ──────────────────────────────────────────
    w = _load_weights()
    c_PM = w.get("c_PM", 1.0); c_CM = w.get("c_CM", 7.0)
    c_fail = w.get("c_fail", 25.0); w_hold = w.get("w_hold", 0.005)

    maint_cost   = n_PM * c_PM + n_CM * c_CM
    failure_cost = n_fail * c_fail
    order_cost   = float(getattr(env, "_episode_order_cost", 0.0))
    avg_inv      = float(env.resource_state.consumable_inventory.sum())
    holding_cost = avg_inv * w_hold * T
    total_cost   = maint_cost + failure_cost + order_cost + holding_cost

    # Inventory fill rate: actual CM / (actual + still queued)
    cm_queued = len(getattr(env, "_cm_queue", set()))
    fill_rate = float(n_CM / max(n_CM + cm_queued, 1))

    # EOQ adherence: how close is reorder_qty to theoretical EOQ?
    # EOQ = sqrt(2*D*K/h), D=demand/episode, K=order cost, h=holding rate
    eoq_ratios = []
    for i, r in enumerate(cfg.get("resources", {}).get("consumable", [])):
        D  = n_CM * float(env.rho_CM[0, env.n_renewable + i])
        K  = float(r.get("reorder_cost", 10.0))
        h  = float(w_hold)
        if D > 0 and h > 0:
            eoq = math.sqrt(2 * D * K / h)
            actual = float(r.get("reorder_qty", 8.0))
            eoq_ratios.append(actual / max(eoq, 1e-8))
    eoq_ratio = float(np.mean(eoq_ratios)) if eoq_ratios else 1.0

    return {
        # JSS
        "jobs_completed":        float(n_done),
        "service_level":         float(service_level),
        "weighted_tardiness":    float(wt_sum),
        "wt_normalised":         float(wt_norm),
        "makespan":              float(makespan),
        "mean_flow_time":        float(mean_flow),
        "throughput":            float(throughput),
        "machine_utilisation":   float(machine_utilisation),
        # Reliability
        "availability":          float(availability),
        "inherent_availability": float(inherent_avail),
        "failures":              float(n_fail),
        "n_PM":                  float(n_PM),
        "n_CM":                  float(n_CM),
        "pm_cm_ratio":           float(pm_cm_ratio),
        "mtbf":                  float(mtbf),
        "mttr":                  float(mttr),
        "mean_health":           float(mean_health),
        "mean_rul_norm":         float(mean_rul_norm),
        "mean_hazard_rate":      float(mean_hazard),
        "health_cv":             float(health_cv),
        # Cost/Inventory
        "maint_cost":            float(maint_cost),
        "failure_cost":          float(failure_cost),
        "order_cost":            float(order_cost),
        "holding_cost":          float(holding_cost),
        "total_cost":            float(total_cost),
        "avg_inventory":         float(avg_inv),
        "fill_rate":             float(fill_rate),
        "eoq_ratio":             float(eoq_ratio),
    }
