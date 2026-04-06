"""
A4_independent_ppo.py  —  Ablation A4: Independent PPO vs MAPPO
================================================================
Proves that cooperative MARL (shared critic + shared failure signal)
outperforms independent PPO (no coordination).

Two training conditions on a SMALL instance (M=3, J=10, T=50 steps)
to get meaningful results in 10k–20k timesteps (~5–15 minutes):

  Condition A — MAPPO (cooperative):
    - Centralised critic sees both agents' observations
    - Shared failure penalty λ=0.3 couples agents
    - Agent 2 TGIN sees machine health features
    → Standard trained policy

  Condition B — Independent PPO:
    - Each agent has its own critic (no centralised critic)
    - λ=0 (no shared failure signal — agents are decoupled)
    - Agent 2 still uses TGIN but no shared failure gradient
    → Simulates solving the two sub-problems independently

Expected:
  MAPPO > IndepPPO on failures/ep and joint metrics.
  This shows the benefit of cooperation.

Usage:
  python ablations/A4_independent_ppo.py \\
      --timesteps 20000 \\
      --outdir results/ablation_a4/

  # With full checkpoint for comparison
  python ablations/A4_independent_ppo.py \\
      --timesteps 20000 \\
      --full-checkpoint checkpoints/phase3_500k.pt \\
      --outdir results/ablation_a4/
"""

import argparse, os, sys, yaml, copy, subprocess
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ablations.ablation_utils import (
    load_config, eval_marl_policy, eval_baselines,
    compare_table, save_results, statistical_summary, AblationResult
)
import numpy as np


# Small instance config for fast retraining
SMALL_INSTANCE_OVERRIDES = {
    "episode": {
        "t_max_train": 50,
        "t_max_eval": 100,
        "dt_hours": 8.0,
    },
    "jobs": {
        "n_jobs_train": 10,
        "n_jobs_eval": 10,
        "n_ops_min": 2,
        "n_ops_max": 4,
        "proc_time_min_hours": 16.0,
        "proc_time_max_hours": 32.0,
    },
    "mappo": {
        "rollout_steps": 256,
        "ppo_epochs": 5,
        "minibatch_size": 32,
        "lr_actor1": 3e-4,
        "lr_actor2": 3e-4,
        "lr_critic": 1e-3,
        "entropy_coef": 0.05,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
    },
    "stochasticity_level": 1,
}


def make_small_config(base_cfg: dict, indep_ppo: bool = False) -> dict:
    """Create a small-instance config, optionally for independent PPO."""
    cfg = copy.deepcopy(base_cfg)

    # Apply small instance overrides
    for section, overrides in SMALL_INSTANCE_OVERRIDES.items():
        if section in cfg:
            cfg[section].update(overrides)
        else:
            cfg[section] = overrides

    # Keep only first 3 machines
    if "machines" in cfg:
        cfg["machines"] = cfg["machines"][:3]
        # Renumber
        for i, m in enumerate(cfg["machines"]):
            m["machine_id"] = i

    # Independent PPO: decouple agents
    if indep_ppo:
        cfg["_independent_ppo"] = True  # flag for trainer
        cfg.setdefault("rewards", {})

    return cfg


def train_small_instance(
    cfg: dict,
    timesteps: int,
    outdir: str,
    run_name: str,
) -> str:
    """Train a policy on the small instance. Returns checkpoint path."""
    import tempfile
    cfg_path = os.path.join(outdir, f"{run_name}_config.yaml")
    ckpt_dir = os.path.join(outdir, run_name, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cmd = [
        sys.executable, str(ROOT / "scripts" / "train.py"),
        "--config",    cfg_path,
        "--timesteps", str(timesteps),
        "--outdir",    os.path.join(outdir, run_name),
    ]
    print(f"  Training {run_name}: {timesteps} steps...")
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)

    if result.returncode != 0:
        print(f"  WARNING: Training failed for {run_name}")
        return None

    ckpt = os.path.join(outdir, run_name, "checkpoints", "latest.pt")
    if os.path.exists(ckpt):
        print(f"  Checkpoint: {ckpt}")
        return ckpt
    return None


def eval_with_indep_ppo_weights(
    checkpoint_path: str,
    cfg: dict,
    n_episodes: int,
    name: str = "IndepPPO",
) -> AblationResult:
    """
    Evaluate a checkpoint trained with λ=0 (independent PPO approximation).
    Uses patch_weights to set λ=0 during evaluation too.
    """
    from ablations.ablation_utils import patch_weights
    with patch_weights(lambda_shared=0.0):
        return eval_marl_policy(
            checkpoint_path=checkpoint_path,
            config=cfg,
            n_episodes=n_episodes,
            stoch_level=cfg.get("stochasticity_level", 1),
            name=name,
        )


def main():
    parser = argparse.ArgumentParser(description="A4: Independent PPO vs MAPPO")
    parser.add_argument("--timesteps",        type=int,  default=20000,
                        help="Training steps for small instance (default: 20000 ≈ 10min)")
    parser.add_argument("--eval-episodes",    type=int,  default=50)
    parser.add_argument("--full-checkpoint",  default=None,
                        help="Optional: full M=5 Phase3 checkpoint for comparison")
    parser.add_argument("--outdir",           default="results/ablation_a4/")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    base_cfg = load_config()

    print("\n" + "="*60)
    print("  ABLATION A4: INDEPENDENT PPO vs MAPPO")
    print(f"  Small instance: M=3, J=10, T=50, {args.timesteps} steps")
    print("="*60)

    # ── Condition A: MAPPO (cooperative, λ=0.3) ──────────────────────────────
    print("\n[Condition A] Training MAPPO (cooperative, λ=0.3)...")
    cfg_mappo = make_small_config(base_cfg, indep_ppo=False)
    ckpt_mappo = train_small_instance(
        cfg=cfg_mappo,
        timesteps=args.timesteps,
        outdir=args.outdir,
        run_name="mappo_cooperative",
    )

    # ── Condition B: Independent PPO (λ=0, decoupled) ───────────────────────
    print("\n[Condition B] Training Independent PPO (λ=0, decoupled)...")
    cfg_indep = make_small_config(base_cfg, indep_ppo=True)
    # Patch reward weights to λ=0 before training
    import contextlib
    from ablations.ablation_utils import patch_weights
    with patch_weights(lambda_shared=0.0):
        ckpt_indep = train_small_instance(
            cfg=cfg_indep,
            timesteps=args.timesteps,
            outdir=args.outdir,
            run_name="indep_ppo_decoupled",
        )

    # ── Evaluation ────────────────────────────────────────────────────────────
    results = []

    if ckpt_mappo and os.path.exists(ckpt_mappo):
        print(f"\n[Eval A] MAPPO checkpoint: {ckpt_mappo}")
        r_mappo = eval_marl_policy(
            checkpoint_path=ckpt_mappo,
            config=cfg_mappo,
            n_episodes=args.eval_episodes,
            stoch_level=1,
            name="MAPPO (cooperative)",
        )
        results.append(r_mappo)
    else:
        print("  MAPPO checkpoint not found — skipping")

    if ckpt_indep and os.path.exists(ckpt_indep):
        print(f"\n[Eval B] IndepPPO checkpoint: {ckpt_indep}")
        r_indep = eval_with_indep_ppo_weights(
            checkpoint_path=ckpt_indep,
            cfg=cfg_indep,
            n_episodes=args.eval_episodes,
            name="IndepPPO (decoupled)",
        )
        results.append(r_indep)
    else:
        print("  IndepPPO checkpoint not found — skipping")

    # ── Full M=5 MARL for reference ───────────────────────────────────────────
    if args.full_checkpoint and os.path.exists(args.full_checkpoint):
        print(f"\n[Eval C] Full M=5 MARL checkpoint: {args.full_checkpoint}")
        r_full = eval_marl_policy(
            checkpoint_path=args.full_checkpoint,
            config=base_cfg,
            n_episodes=args.eval_episodes,
            stoch_level=3,
            name="MARL_M5_Phase3",
        )
        results.append(r_full)

    # ── Baselines on small instance ───────────────────────────────────────────
    print("\n[Baselines] Evaluating baselines on small instance...")
    baseline_results = eval_baselines(
        config=cfg_mappo,
        n_episodes=args.eval_episodes,
        stoch_level=1,
    )
    results.extend(baseline_results)

    if len(results) < 2:
        print("\n  Not enough results to compare. Check training logs.")
        return

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n--- COMPARISON TABLE ---")
    kpis = ["failures", "n_PM", "n_CM", "completions", "service_level", "avg_health"]
    table = compare_table(results, kpis=kpis)

    # Statistical test
    r_a = next((r for r in results if "MAPPO" in r.name and "cooperative" in r.name), None)
    r_b = next((r for r in results if "Indep" in r.name), None)
    if r_a and r_b:
        print("\n--- STATISTICAL TEST: MAPPO vs IndepPPO ---")
        stats = statistical_summary(r_a, r_b, kpis=kpis)
        for kpi, s in stats.items():
            sig = "* p<0.05 *" if s["significant"] else ""
            print(f"  {kpi:<20}: MAPPO={s['mean_a']:.3f}  "
                  f"IndepPPO={s['mean_b']:.3f}  p={s['p_welch']:.4f}  {sig}")

    save_results(results, args.outdir, "a4_indep_vs_mappo")

    import json
    report = {
        "ablation": "A4_independent_ppo_vs_mappo",
        "timesteps": args.timesteps,
        "instance": "M=3, J=10, T=50",
        "results": {r.name: r.means for r in results},
        "note": (
            f"Small instance ({args.timesteps} steps). "
            "MAPPO uses centralised critic + λ=0.3 shared penalty. "
            "IndepPPO uses λ=0 (decoupled agents, independent critics)."
        )
    }
    with open(os.path.join(args.outdir, "a4_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.outdir, "a4_table.txt"), "w") as f:
        f.write(table)

    print(f"\n  Outputs saved to: {args.outdir}")
    print(f"  Note: Small instance ({args.timesteps} steps) — qualitative comparison only.")
    print(f"  For publication-quality results: run 500k steps full instance.")


if __name__ == "__main__":
    main()
