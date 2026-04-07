"""
ablation_utils.py
=================
Shared utilities for all ablation studies.

Provides:
  - eval_marl_policy()   : run N episodes with trained MARL agents and collect KPIs
  - eval_baselines()     : run N episodes with all 4 baselines
  - compare_table()      : print/return a comparison table
  - patch_weights()      : context manager to temporarily change reward weights
  - AblationResult       : dataclass for storing results
"""

import os, sys, copy, contextlib, json, yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── KPIs collected per episode ─────────────────────────────────────────────────
EPISODE_KPIS = [
    "failures", "n_PM", "n_CM", "pm_cm_ratio",
    "completions", "tardiness", "service_level", "avg_health",
]

HIGHER_BETTER = {
    "failures": False, "n_PM": True, "n_CM": False, "pm_cm_ratio": True,
    "completions": True, "tardiness": False, "service_level": True,
    "avg_health": True,
}


# ── Result container ───────────────────────────────────────────────────────────
@dataclass
class AblationResult:
    name: str
    kpi_data: Dict[str, List[float]]  # {kpi_name: [values per episode]}

    @property
    def means(self) -> Dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self.kpi_data.items() if v}

    @property
    def stds(self) -> Dict[str, float]:
        return {k: float(np.std(v)) for k, v in self.kpi_data.items() if v}

    def to_dict(self) -> dict:
        return {"name": self.name, "kpi_data": {k: list(v) for k, v in self.kpi_data.items()}}


# ── Config helpers ─────────────────────────────────────────────────────────────
def load_config(config_path: str = "configs/base.yaml") -> dict:
    path = ROOT / config_path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_reward_weights(weights_path: str = "rewards/reward_weights.yaml") -> dict:
    path = ROOT / weights_path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_reward_weights(weights: dict, path: str = "rewards/reward_weights.yaml"):
    full_path = ROOT / path
    with open(full_path, "w", encoding="utf-8") as f:
        yaml.dump(weights, f, default_flow_style=False)


@contextlib.contextmanager
def patch_weights(**overrides):
    """
    Context manager: temporarily overrides reward weights, restores on exit.

    Usage:
        with patch_weights(w_RUL=0.0, lambda_shared=0.0):
            result = eval_marl_policy(...)
    """
    path = ROOT / "rewards" / "reward_weights.yaml"
    with open(path, encoding="utf-8") as f:
        original_text = f.read()
    original = yaml.safe_load(original_text)

    patched = copy.deepcopy(original)
    patched.update(overrides)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(patched, f, default_flow_style=False)

    changed = {k: (original.get(k), v) for k, v in overrides.items()}
    print(f"  [patch_weights] Applied: {changed}")

    try:
        yield patched
    finally:
        with open(path, "w", encoding="utf-8") as f:
            f.write(original_text)
        print(f"  [patch_weights] Restored original weights")


# ── Run one episode with baseline policy ───────────────────────────────────────
def _run_baseline_episode(env, baseline, seed: int) -> dict:
    """Run one episode with a BaselinePolicy and return KPI dict."""
    from environments.mfg_env import AGENT_PDM
    baseline.reset()
    env.reset(seed=seed)
    done, steps, n_PM, n_CM = False, 0, 0, 0

    while not done and steps < 300:
        a1 = baseline.agent1_action(env)
        env._step_agent1(a1)
        a2 = baseline.agent2_action(env)
        env._step_agent2(a2)
        env._resolve_physics()
        env._compute_rewards()

        n_PM += sum(1 for a in a1["maintenance"] if a == 1)
        pass  # n_CM counted from env._episode_cm at episode end
        done = env.terminations[AGENT_PDM] or env.truncations[AGENT_PDM]
        steps += 1

    completed = [j for j in env.jobs if j.completion_time is not None]
    on_time   = [j for j in completed if j.tardiness == 0]
    tard      = sum(j.weight * j.tardiness for j in completed)
    return {
        "failures":      env._episode_failures,
        "n_PM":          n_PM,
        "n_CM":          getattr(env, "_episode_cm", 0),  # auto-CM count
        "pm_cm_ratio":   n_PM / max(getattr(env, "_episode_cm", 1), 1),
        "completions":   len(completed),
        "tardiness":     float(tard),
        "service_level": len(on_time) / max(len(completed), 1),
        "avg_health":    float(np.mean([s.health for s in env.machine_states])),
    }


# ── Run one episode with MARL agents ──────────────────────────────────────────
def _run_marl_episode(env, agent1, agent2, seed: int) -> dict:
    """Run one episode with trained MARL agents and return KPI dict."""
    from environments.mfg_env import AGENT_PDM
    env.reset(seed=seed)
    done, steps, n_PM, n_CM = False, 0, 0, 0

    while not done and steps < 300:
        # Agent 1 action
        obs1 = env._build_agent1_obs()
        action1, _, _ = agent1.act(
            obs_np=obs1,
            machine_states=env.machine_states,
            machine_busy=env.machine_busy,
            resource_state=env.resource_state,
            rho_PM=env.rho_PM,
            rho_CM=env.rho_CM,
        )
        env._step_agent1(action1)
        n_PM += sum(1 for a in action1["maintenance"] if a == 1)
        pass  # n_CM counted from env._episode_cm at episode end

        # Agent 2 action
        obs2, valid_pairs = env._build_agent2_obs()
        if valid_pairs:
            _, idx, _, _ = agent2.act(obs2, valid_pairs)
            env._step_agent2(idx)
        else:
            env._step_agent2(len(valid_pairs))  # WAIT

        env._resolve_physics()
        env._compute_rewards()
        done = env.terminations[AGENT_PDM] or env.truncations[AGENT_PDM]
        steps += 1

    completed = [j for j in env.jobs if j.completion_time is not None]
    on_time   = [j for j in completed if j.tardiness == 0]
    tard      = sum(j.weight * j.tardiness for j in completed)
    return {
        "failures":      env._episode_failures,
        "n_PM":          n_PM,
        "n_CM":          getattr(env, "_episode_cm", 0),  # auto-CM count
        "pm_cm_ratio":   n_PM / max(getattr(env, "_episode_cm", 1), 1),
        "completions":   len(completed),
        "tardiness":     float(tard),
        "service_level": len(on_time) / max(len(completed), 1),
        "avg_health":    float(np.mean([s.health for s in env.machine_states])),
    }


# ── Top-level evaluation functions ─────────────────────────────────────────────
def eval_marl_policy(
    checkpoint_path: str,
    config: dict,
    n_episodes: int = 50,
    stoch_level: int = 3,
    seed_offset: int = 42,
    name: str = "MARL",
) -> AblationResult:
    """
    Load a MARL checkpoint and evaluate for n_episodes.
    Returns AblationResult with per-episode KPIs.
    """
    from environments.mfg_env import ManufacturingEnv
    from agents.pdm_agent import PDMAgent
    from agents.jobshop_agent import JobShopAgent
    from models.critic import CentralizedCritic
    from utils.checkpoint import load_checkpoint
    import torch

    cfg = copy.deepcopy(config)
    cfg["stochasticity_level"] = stoch_level

    env    = ManufacturingEnv(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Infer obs_dim from env
    env.reset(seed=0)
    obs1_sample = env._build_agent1_obs()
    actual_obs_dim = len(obs1_sample)

    agent1 = PDMAgent(cfg, device=device, obs_dim=actual_obs_dim)
    agent2 = JobShopAgent(cfg, device=device)
    critic = CentralizedCritic(cfg)

    load_checkpoint(checkpoint_path, agent1, agent2, critic, device=device)
    agent1.eval(); agent2.eval(); critic.eval()

    kpi_data = {k: [] for k in EPISODE_KPIS}

    print(f"  Evaluating {name} ({n_episodes} episodes, stoch={stoch_level})...")
    for ep in range(n_episodes):
        m = _run_marl_episode(env, agent1, agent2, seed=seed_offset + ep * 7)
        for k in EPISODE_KPIS:
            kpi_data[k].append(m[k])
        sys.stdout.write(f"\r    Episode {ep+1}/{n_episodes}")
        sys.stdout.flush()
    print()

    return AblationResult(name=name, kpi_data=kpi_data)


def eval_baselines(
    config: dict,
    n_episodes: int = 50,
    stoch_level: int = 3,
    seed_offset: int = 42,
) -> List[AblationResult]:
    """Evaluate all 4 baseline policies."""
    from environments.mfg_env import ManufacturingEnv
    from benchmarks.baselines import get_all_baselines

    cfg = copy.deepcopy(config)
    cfg["stochasticity_level"] = stoch_level
    env = ManufacturingEnv(cfg)
    baselines = get_all_baselines()
    results = []

    for b in baselines:
        kpi_data = {k: [] for k in EPISODE_KPIS}
        print(f"  Baseline: {b.name} ({n_episodes} episodes)...")
        for ep in range(n_episodes):
            m = _run_baseline_episode(env, b, seed=seed_offset + ep * 7)
            for k in EPISODE_KPIS:
                kpi_data[k].append(m[k])
            sys.stdout.write(f"\r    Episode {ep+1}/{n_episodes}")
            sys.stdout.flush()
        print()
        results.append(AblationResult(name=b.name, kpi_data=kpi_data))

    return results


# ── Comparison table ──────────────────────────────────────────────────────────
def compare_table(
    results: List[AblationResult],
    kpis: Optional[List[str]] = None,
    reference_name: Optional[str] = None,
) -> str:
    """
    Print and return a formatted comparison table.

    Args:
        results:        List of AblationResult to compare
        kpis:           Which KPIs to include (default: all)
        reference_name: Name of reference condition (e.g. "MARL_full")
                        If set, p-values vs this condition are shown.
    """
    if kpis is None:
        kpis = EPISODE_KPIS

    # Build reference data for p-values
    ref_data = None
    if reference_name:
        for r in results:
            if r.name == reference_name:
                ref_data = r.kpi_data
                break

    # Header
    col_w = 14
    name_w = 22
    header = f"{'KPI':<20}"
    for r in results:
        header += f"{r.name[:col_w]:>{col_w}}"
    if ref_data:
        header += f"  {'p-val vs ' + reference_name[:8]:>16}"
    lines = ["=" * (20 + col_w * len(results) + 20), header,
             "-" * (20 + col_w * len(results) + 20)]

    # Rows
    for kpi in kpis:
        row = f"{kpi:<20}"
        means = []
        for r in results:
            if kpi in r.kpi_data and r.kpi_data[kpi]:
                m = np.mean(r.kpi_data[kpi])
                s = np.std(r.kpi_data[kpi])
                cell = f"{m:.2f}±{s:.2f}"
                means.append((r.name, m, r.kpi_data[kpi]))
            else:
                cell = "N/A"
                means.append((r.name, None, []))
            row += f"{cell:>{col_w}}"

        # Star best
        valid = [(n, m, d) for n, m, d in means if m is not None]
        if valid:
            if HIGHER_BETTER.get(kpi, True):
                best_name = max(valid, key=lambda x: x[1])[0]
            else:
                best_name = min(valid, key=lambda x: x[1])[0]
            row = row.replace(f"{best_name[:col_w]:>{col_w}}", f"{'★' + best_name[:col_w-1]:>{col_w}}")

        # p-value vs reference
        if ref_data and kpi in ref_data and ref_data[kpi]:
            ref_vals = ref_data[kpi]
            p_str = ""
            for n, m, d in means:
                if n != reference_name and d:
                    try:
                        _, p = stats.ttest_ind(ref_vals, d)
                        p_str = f"{'<0.001' if p < 0.001 else f'{p:.3f}':>8}"
                    except:
                        p_str = "  N/A  "
                    break
            row += f"  {p_str:>16}"

        lines.append(row)

    lines.append("=" * (20 + col_w * len(results) + 20))
    table = "\n".join(lines)
    print(table)
    return table


# ── Save/load results ─────────────────────────────────────────────────────────
def save_results(results: List[AblationResult], outdir: str, name: str = "results"):
    """Save results to JSON."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.json")
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"  Saved: {path}")


def load_results(path: str) -> List[AblationResult]:
    with open(path) as f:
        data = json.load(f)
    return [AblationResult(name=d["name"], kpi_data=d["kpi_data"]) for d in data]


# ── Statistical summary ───────────────────────────────────────────────────────
def statistical_summary(
    result_a: AblationResult,
    result_b: AblationResult,
    kpis: Optional[List[str]] = None,
) -> dict:
    """
    Compute Welch t-test and Cohen's d between two conditions.
    Returns dict: {kpi: {mean_a, mean_b, p_t, cohen_d, significant}}
    """
    if kpis is None:
        kpis = EPISODE_KPIS

    summary = {}
    for kpi in kpis:
        a = np.array(result_a.kpi_data.get(kpi, []))
        b = np.array(result_b.kpi_data.get(kpi, []))
        if len(a) < 2 or len(b) < 2:
            continue
        _, p_t = stats.ttest_ind(a, b)
        pooled_std = np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)
        d = (np.mean(a) - np.mean(b)) / (pooled_std + 1e-10)
        summary[kpi] = {
            "mean_a":      round(float(np.mean(a)), 4),
            "mean_b":      round(float(np.mean(b)), 4),
            "std_a":       round(float(np.std(a)), 4),
            "std_b":       round(float(np.std(b)), 4),
            "p_welch":     round(float(p_t), 4),
            "cohen_d":     round(float(d), 4),
            "significant": bool(p_t < 0.05),
        }
    return summary