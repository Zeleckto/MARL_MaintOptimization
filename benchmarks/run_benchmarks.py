"""
benchmarks/run_benchmarks.py
==============================
Runs all 4 comparison baselines and prints a results table.

Usage:
    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --episodes 10
    python benchmarks/run_benchmarks.py --episodes 5 --seed 42

Output: comparison table with mean ± std for each metric across episodes.

Metrics reported:
  failures      — total machine failures per episode
  PM events     — PM actions taken
  CM events     — CM actions taken
  PM/CM ratio   — higher is better (more proactive)
  completions   — jobs completed
  tardiness     — weighted tardiness (lower is better)
  service level — fraction of jobs on time (higher is better)
  avg health    — mean machine health at episode end

GATE: If ABR+MDD beats all 3 other baselines on ≥2 metrics,
      and MARL (if checkpoint provided) beats ABR+MDD on ≥3 metrics,
      the comparison is meaningful for the paper.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import yaml
from typing import List, Dict

from environments.mfg_env import ManufacturingEnv, AGENT_PDM, AGENT_JOBSHOP
from benchmarks.baselines import get_all_baselines, BaselinePolicy


def run_one_episode(env: ManufacturingEnv, baseline: BaselinePolicy, seed: int) -> Dict:
    """Run one episode and return a metrics dict."""
    baseline.reset()
    env.reset(seed=seed)
    done  = False
    steps = 0
    r1_sum = 0.0
    r2_sum = 0.0
    n_PM = 0
    n_CM = 0

    while not done and steps < 300:
        a1 = baseline.agent1_action(env)
        env._step_agent1(a1)

        a2_idx = baseline.agent2_action(env)
        env._step_agent2(a2_idx)
        env._resolve_physics()
        env._compute_rewards()

        r1_sum += env.rewards[AGENT_PDM]
        r2_sum += env.rewards[AGENT_JOBSHOP]
        n_PM += sum(1 for a in a1["maintenance"] if a == 1)
        n_CM += sum(1 for a in a1["maintenance"] if a == 2)

        done = env.terminations[AGENT_PDM] or env.truncations[AGENT_PDM]
        steps += 1

    # Compute metrics
    completed = [j for j in env.jobs if j.completion_time is not None]
    on_time   = [j for j in completed if j.tardiness == 0]
    tard      = sum(j.weight * j.tardiness for j in completed)
    avg_health = np.mean([s.health for s in env.machine_states])

    return {
        "failures":      env._episode_failures,
        "n_PM":          n_PM,
        "n_CM":          n_CM,
        "pm_cm_ratio":   n_PM / max(n_CM, 1),
        "completions":   len(completed),
        "tardiness":     tard,
        "service_level": len(on_time) / max(len(completed), 1),
        "avg_health":    avg_health,
        "return1":       r1_sum,
        "return2":       r2_sum,
        "steps":         steps,
    }


def run_baseline(env, baseline, n_episodes, seed_offset):
    """Run n_episodes for one baseline. Returns list of metric dicts."""
    results = []
    for ep in range(n_episodes):
        m = run_one_episode(env, baseline, seed=seed_offset + ep * 7)
        results.append(m)
        sys.stdout.write(f"\r    Episode {ep+1}/{n_episodes} ...")
        sys.stdout.flush()
    print()
    return results


def summarise(results: List[Dict]) -> Dict:
    """Returns mean ± std for each metric."""
    keys = list(results[0].keys())
    return {
        k: (np.mean([r[k] for r in results]), np.std([r[k] for r in results]))
        for k in keys
    }


def print_table(summaries: Dict[str, Dict]):
    """Prints comparison table."""
    baselines = list(summaries.keys())
    metrics = [
        ("failures",      "Failures/ep",   "lower"),
        ("n_PM",          "PM events",     "higher"),
        ("n_CM",          "CM events",     "lower"),
        ("pm_cm_ratio",   "PM/CM ratio",   "higher"),
        ("completions",   "Completions",   "higher"),
        ("tardiness",     "Wt. Tardiness", "lower"),
        ("service_level", "Service Level", "higher"),
        ("avg_health",    "Avg Health",    "higher"),
    ]

    # Column widths
    col_w = 22
    name_w = 22

    print()
    print("=" * (name_w + col_w * len(baselines) + 4))
    print("  BASELINE COMPARISON RESULTS")
    print("=" * (name_w + col_w * len(baselines) + 4))
    print()

    # Header
    header = f"  {'Metric':<{name_w}}"
    for b in baselines:
        short = b[:col_w-2]
        header += f"  {short:^{col_w-2}}"
    print(header)
    print("  " + "-" * (name_w + col_w * len(baselines)))

    for key, label, direction in metrics:
        row = f"  {label:<{name_w}}"
        values = [(summaries[b][key][0], summaries[b][key][1]) for b in baselines]
        # Find best
        means = [v[0] for v in values]
        best_mean = min(means) if direction == "lower" else max(means)

        for mean, std in values:
            cell = f"{mean:.2f}±{std:.2f}"
            marker = " *" if abs(mean - best_mean) < 1e-6 else "  "
            row += f"  {cell + marker:^{col_w}}"
        print(row)

    print("  " + "-" * (name_w + col_w * len(baselines)))
    print("  (* = best for this metric)")
    print()

    # Quick summary: how many metrics does ABR+MDD win?
    abr_name = "ABR + MDD + (Q,R)"
    if abr_name in summaries:
        abr_wins = 0
        for key, label, direction in metrics:
            abr_mean = summaries[abr_name][key][0]
            all_means = [summaries[b][key][0] for b in baselines]
            best = min(all_means) if direction == "lower" else max(all_means)
            if abs(abr_mean - best) < 1e-6:
                abr_wins += 1
        print(f"  ABR+MDD wins {abr_wins}/{len(metrics)} metrics.")
        if abr_wins >= 2:
            print("  ✓ ABR+MDD is a meaningful upper bound for MARL comparison.")
        else:
            print("  ⚠  ABR+MDD not clearly dominant — check instance difficulty.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run all 4 baselines")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--jobs",     type=int, default=None, help="override n_jobs")
    args = parser.parse_args()

    print()
    print("Loading config...")
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)

    if args.jobs:
        config["jobs"]["n_jobs_train"] = args.jobs

    env      = ManufacturingEnv(config)
    baselines = get_all_baselines()
    summaries = {}

    for baseline in baselines:
        print(f"  Running: {baseline.name}  ({args.episodes} episodes)")
        results = run_baseline(env, baseline, args.episodes, args.seed)
        summaries[baseline.name] = summarise(results)

    print_table(summaries)


if __name__ == "__main__":
    main()
