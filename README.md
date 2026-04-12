<h1 align="center">Multi-Agent Reinforcement Learning for Joint<br>Predictive Maintenance & Flexible Job Shop Scheduling</h1>

<p align="center">
  <b>BTP2 (MCD412) — Indian Institute of Technology Delhi</b><br>
  Shreenath Jha &amp; Kirtan Gehlot &nbsp;|&nbsp; Supervisor: Dr. Minakshi Kumari<br>
  <i>Department of Mechanical Engineering — April 2026</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?logo=pytorch" />
  <img src="https://img.shields.io/badge/PettingZoo-AEC-green" />
  <img src="https://img.shields.io/badge/Tests-83_passing-brightgreen" />
</p>

---

## Overview

This project addresses the **joint optimisation of Predictive Maintenance (PDM) and Flexible Job Shop Scheduling (FJSP)** — an open problem identified by [Cassady & Kutanoglu (2005)](https://doi.org/10.1109/TR.2005.847270) where independent optimisation of maintenance and scheduling leaves 15–30% of total cost savings on the table.

We implement a cooperative **Multi-Agent Proximal Policy Optimisation (MAPPO)** system with two specialised agents that learn to coordinate maintenance timing with production scheduling in a stochastic manufacturing environment featuring Weibull degradation, Kijima Type I imperfect repair, and shared resources.

### Key Results

| Metric | MARL (Ours) | Reactive+FCFS | Fixed-PM+SPT | ABR+MDD |
|--------|:-----------:|:-------------:|:------------:|:-------:|
| **Jobs completed** | 24.3 | 25.6 | **26.6** | 23.2 |
| **Failures/episode** | **2.7** | 5.1 | **2.7** | 3.6 |
| **Fleet health** | **81.0%** | 60.8% | 77.3% | 75.2% |
| **On-time delivery** | **12.7** | 11.6 | 11.1 | 11.5 |
| **Tardiness** | 989 | 1285 | 1511 | **838** |
| **MTBF (shifts)** | **64** | 31 | 37 | 32 |

**MARL wins 5 of 7 primary metrics** and Pareto-dominates ABR+MDD on all five Pareto metrics. It achieves **47% fewer failures** than reactive maintenance while maintaining 95% of best-baseline throughput.

> All results evaluated across 3 independent seed sets (90 total episodes) with Bonferroni-corrected statistical testing.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│              Manufacturing Environment (PettingZoo AEC)          │
│  5 machines (Weibull degradation)  ·  40 jobs  ·  150 shifts     │
│  Kijima Type I repair (q=0.5)  ·  3 consumable + 3 renewable    │
└──────────┬──────────────────────────────┬────────────────────────┘
           │                              │
 ┌─────────▼──────────┐       ┌──────────▼───────────────────┐
 │  Agent 1 (PDM)     │       │  Agent 2 (Scheduling)        │
 │  MLP Policy        │       │  TGIN + Hybrid ActionScorer  │
 │                    │       │                              │
 │  Obs: health, RUL, │       │  Obs: heterogeneous graph    │
 │       inventory    │       │       + raw env state        │
 │                    │       │                              │
 │  Action: binary PM │       │  Action: select (j, op, m)   │
 │    per machine +   │       │    from valid pairs or WAIT  │
 │    reorder qty     │       │                              │
 │                    │       │  Scoring:                    │
 │  Constraint:       │       │    Linear (frozen, expert)   │
 │    h < 75% gate    │       │    + MLP residual (learned)  │
 └─────────┬──────────┘       └──────────┬───────────────────┘
           │         Shared Reward        │
           │    R = -40·fail + 1·comp     │
           │         (λ = 0.5 each)       │
           └──────────┬──────────────────┘
                ┌─────▼──────┐
                │   Critic   │
                │ Centralised│
                │ (trained   │
                │  on r₁)    │
                └────────────┘
```

### Agent 1: Predictive Maintenance

- **Architecture:** MLP (85 → 256 → 256 → heads)
- **Observation:** Machine health, virtual age, RUL estimates, resource inventory, job summary statistics
- **Action:** Binary PM decision per machine (masked by h < 75% gate) + continuous reorder quantities
- **Key result:** Learned PM timing reduces failures from 5.1 → 2.7 (47% reduction)

### Agent 2: Job Shop Scheduling

- **Architecture:** TGIN graph encoder + Hybrid ActionScorer
- **TGIN:** 3-layer Graph Isomorphism Network processing heterogeneous manufacturing graph (operation/machine/job nodes)
- **ActionScorer (v4 — final):** Two-head scorer on 7 hand-crafted features:

```
score_k = w_frozen · f_k + b_frozen   +   MLP_θ(f_k)
          ─────────────────────────       ────────────
          Linear head (FROZEN)             Residual (trainable, lr=3e-5)
          Expert fewest-ops-left           Learns corrections
```

**7 hand-crafted features per valid (job, operation, machine) pair:**

| # | Feature | Range | What it encodes |
|---|---------|-------|-----------------|
| 0 | remaining_ops / total | [0, 1] | Job proximity to completion |
| 1 | processing_time / 10 | [0, 1] | Operation duration |
| 2 | slack / T_max | [-1, 1] | Time buffer to due date |
| 3 | machine_health / 100 | [0, 1] | Machine condition |
| 4 | is_last_op | {0, 1} | Will this complete the job? |
| 5 | progress | [0, 1] | Fraction of ops completed |
| 6 | urgency | [-2, 2] | -slack / proc_time |

**Expert weights:** `w = [-2.5, -3.5, 0.5, 0.5, 3.0, 2.0, 0.5]` — blends fewest-ops-left with SPT, optimised via grid search over 30 episodes.

### Centralised Critic

Trained on Agent 1 returns (r₁). Agent 2 uses a running-mean baseline `v₂ = EMA(r₂)` instead of the shared critic, because the critic predicts maintenance value (r₁) which is uninformative for scheduling advantages.

---

## Environment

| Parameter | Value |
|-----------|-------|
| Machines | 5 heterogeneous (CNC Mill, Lathe, Grinder, Press, CMM) |
| Degradation | Weibull (β=2.2–3.2, η=700–1350h, MTBF=78–150 shifts) |
| Repair model | Kijima Type I, q=0.5 (imperfect, diminishing returns) |
| Jobs | 40 (batch at t=0), 2–6 ops each, ~166 total operations |
| Processing time | 2–8 shifts per operation |
| Horizon | 150 shifts |
| Work/capacity ratio | 91.6% (heavily loaded) |
| Resources | 3 consumable (spare parts, lubricants, tooling) + 3 renewable |
| PM duration | 2 shifts, CM duration: 6 shifts |
| Health gate | PM blocked if health > 75% |

Weibull parameters grounded in industrial reliability literature (Mobley 2002, Jardine & Tsang 2006, Ebeling 2013).

---

## Reward Structure (v5 — Final)

**Shared:**
```
R_shared = -40 × n_failures + 1.0 × n_completions    (λ=0.5 to each agent)
```

**Agent 1 (Maintenance):**
```
r₁ = -1.0×PM - 7.0×CM - 0.05×ordering - 0.005×inventory + 0.5×R_shared
```

**Agent 2 (Scheduling):**
```
r₂ = +0.5×assign - 0.3×wait + 5.0×jobs - 8.0×tard/T + 0.5×health + 0.5×R_shared
```

The reward underwent **6 major revisions**. Key removals: ΔRUL (33% noise, SNR=0.48), PM bonus (caused 120 PMs/ep), availability (punished PM), stockout (catch-22 with PM). See thesis Section 5.4 for full evolution.

---

## Training Pipeline

### Phase 1 (Primary): 225k steps

| Phase | Steps | Jobs | Failures | Key Event |
|-------|-------|------|----------|-----------|
| Exploration | 0–30k | 18.9 | 4.3 | Random policy |
| Agent 1 learning | 30–100k | 19.8 | 3.2 | PM timing learned |
| Convergence | 100–200k | 19.5 | 3.1 | Failures stabilised |
| **ActionScorer init** | **200k** | **→ 24.3** | **2.7** | **Expert weights loaded** |
| Stable | 200–225k | 24.3 | 2.7 | ✅ Final checkpoint |

### Behavioural Cloning Experiment

Before adopting expert weight initialisation, we attempted behavioural cloning (BC) to warm-start Agent 2:

1. **Expert data:** 50 episodes of fewest-ops-left scheduling (28.2 jobs/ep), ~5,750 samples
2. **BC training:** Cross-entropy loss on Agent 2's full forward pass (TGIN + ActionScorer)

| Architecture | Accuracy | Random baseline |
|-------------|----------|-----------------|
| TGIN-only scorer | 15.2% | ~10% |
| TGIN-only (higher LR) | 15.8% | ~10% |
| Hybrid (features + TGIN) | 37.3% | ~10% |
| Features-only (fast_mlp) | 37.1% | ~10% |

**Failure analysis:**
- **TGIN-only (15%):** GIN aggregation (sum over neighbours) destroys the "remaining operations" count. Information bottleneck in graph convolution.
- **Hybrid (37%):** Near theoretical ceiling — when 3 valid pairs all have 2 remaining ops, expert picks one but BC penalises the equally-good alternatives.

**Conclusion:** BC was abandoned in favour of direct linear weight initialisation. The expert policy (fewest-ops-left) reduces to 7 weights — no learning needed.

### ActionScorer Evolution

| Version | Architecture | Jobs | Problem |
|---------|-------------|------|---------|
| v1 | TGIN embeddings → MLP | 19–21 | TGIN can't encode scheduling features |
| v2 | Features + TGIN hybrid | ~20 | RL still overrode features |
| v3 | Pure feature MLP, expert init | 24–27 | RL corrupted weights in 200 eps |
| **v4** | **Linear (frozen) + MLP residual** | **24.3** | **✅ Stable — frozen expert + learned corrections** |

### RL Degradation Prevention

Without freezing, the MLP residual head grows to magnitude ~4 within 600 RL episodes, overriding the frozen linear head (magnitude ~5). Jobs drop from 24 → 18. The frozen linear head guarantees a performance floor.

---

## Bug Registry

**19 bugs** found and fixed across 4 categories. The most critical:

| Bug | Impact | Fix |
|-----|--------|-----|
| **E4:** tick_all PM bypass | 238 PMs at h=100% | PM exclusively in _step_agent1 |
| **E5:** Phantom -5/step penalty | r₁ off by -750/ep | Store applied (not requested) PMs |
| **T2:** PPO mask mismatch | entropy stuck at 3.09 for 120k steps | Store mask in rollout buffer |
| **T5:** ActionScorer not saved | Reset to random on every resume | Include in checkpoint |
| **A1:** TGIN can't encode features | BC accuracy 15% (random) | Hand-crafted features |
| **R5:** ΔRUL noise | 33% of r₁ was noise (SNR=0.48) | w_RUL = 0 |

Full registry with root causes in thesis Appendix A.

---

## Setup

### Prerequisites

- Python 3.11+
- CUDA 12.1+ (for GPU training) or CPU-only
- ~4 GB disk space (checkpoints + TensorBoard logs)

### Installation

```bash
# Clone
git clone https://github.com/<your-username>/manufacturing_marl.git
cd manufacturing_marl

# Virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Core dependencies
pip install -r requirements.txt

# PyTorch Geometric (match your CUDA version)
# CUDA 12.1:
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
# CPU only:
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

# Verify installation
python -m pytest tests/ -q --ignore=tests/test_degradation.py \
    --ignore=tests/test_graph_builder.py --ignore=tests/test_gae.py
python test_resources_exhaustive.py
```

Expected: `55 passed` + `ALL RESOURCE DYNAMICS TESTS PASSED`.

---

## Usage

### Training

```bash
# Phase 1: Full training from scratch (500k steps ≈ 3333 episodes)
python scripts/train.py --config configs/phase1.yaml --timesteps 500000

# Resume from checkpoint (linear head auto-frozen, Agent 1 trainable)
python scripts/train.py --config configs/phase1.yaml --timesteps 400000 \
    --resume outputs/checkpoints/phase1_step_0200k.pt

# Monitor training
tensorboard --logdir outputs/runs/
```

**On resume:** The script automatically freezes the ActionScorer linear head (preventing RL degradation), rebuilds Agent 2's optimizer with only trainable parameters, and skips loading Agent 2's optimizer state (architecture mismatch).

### Evaluation

```bash
# MARL vs all baselines (3 seed sets × 30 episodes = 90 total)
python scripts/eval_checkpoint.py \
    --checkpoint outputs/checkpoints/phase1_step_0225k.pt \
    --seed-sets 3 --episodes 30 \
    --outdir outputs/eval_phase1
```

### Ablation Study

```bash
# Full ablation suite: 11 ablations + sensitivity sweeps + baselines + stats
python scripts/run_all_ablations.py \
    --checkpoint outputs/checkpoints/phase1_step_0225k.pt \
    --episodes 30
```

Runs 25+ configurations:

| ID | Ablation | Tests |
|----|----------|-------|
| A0 | MARL baseline | Reference |
| A1 | No shared reward (λ=0) | Cooperation value |
| A2 | No health gate | PM constraint necessity |
| A3 | No assignment bonus | Dense signal value |
| A4 | No PM (Agent 1 off) | PM learning contribution |
| A5 | FCFS scheduling | Scheduling learning vs FCFS |
| A6 | Random scheduling | Lower bound |
| A7 | SPT scheduling | Standard heuristic |
| A8 | Both random | Absolute lower bound |
| A9 | Tight inventory (inv=15) | Phase 2 proxy |
| A10 | Stochastic processing | Phase 3 proxy |
| A11 | λ sweep (0.0–1.0) | Coupling sensitivity |
| A12 | Zero-shot M=5→10 | Scalability |
| A13 | c_fail sweep (10–60) | Failure cost sensitivity |
| A14 | c_PM sweep (0.5–5.0) | PM cost sensitivity |

Output: `outputs/ablation_study/` with Excel, CSVs, 3 plot types, statistical tests, auto-generated report.

### Analysis

```bash
# Training curves (pre-BC vs post-BC impact analysis)
python scripts/training_impact_analysis.py --outdir outputs/results/training

# Training analysis from TensorBoard
python scripts/analyze_training.py \
    --runs outputs/runs/phase1_XXXXXXX outputs/runs/phase1_YYYYYYY \
    --outdir outputs/results/training

# Statistical inference (reads ablation CSVs, no re-evaluation)
python scripts/statistical_inference.py \
    --datadir outputs/ablation_study \
    --outdir outputs/statistical_analysis

# Identify TensorBoard run folders
python scripts/identify_runs.py --rundir outputs/runs/

# Resource inventory analysis (sawtooth plots)
python analyze_resources.py --seed 42

# Regenerate ablation plots from saved CSV
python scripts/regenerate_plots.py --datadir outputs/ablation_study
```

### Tests

```bash
# Unit tests (55 tests covering rewards, masking, inventory, Kijima)
python -m pytest tests/ -q --ignore=tests/test_degradation.py \
    --ignore=tests/test_graph_builder.py --ignore=tests/test_gae.py

# Resource dynamics tests (28 tests: PM/CM consumption, sawtooth, ordering)
python test_resources_exhaustive.py
```

---

## Project Structure

```
manufacturing_marl/
│
├── configs/
│   ├── base.yaml                 # Environment + hyperparameters
│   └── phase1.yaml               # Phase 1 config
│
├── environments/
│   ├── mfg_env.py                # PettingZoo AEC environment (main file)
│   ├── spaces/
│   │   ├── action_spaces.py      # Action masking (h<75% PM gate)
│   │   └── observation_spaces.py # Observation space definitions
│   └── transitions/
│       ├── degradation.py        # Weibull degradation + Kijima repair
│       ├── job_dynamics.py       # Job/operation processing logic
│       ├── resource_dynamics.py  # Consumable + renewable resource management
│       └── failure_handler.py    # Stochastic failure + CM triggering
│
├── agents/
│   ├── pdm_agent.py              # Agent 1: MLP policy for PM + reorder
│   └── jobshop_agent.py          # Agent 2: TGIN + ActionScorer wrapper
│
├── models/
│   ├── critic.py                 # Centralised critic (trained on r₁)
│   ├── mlp_policy.py             # MLP policy (used by Agent 1)
│   └── tgin/
│       ├── tgin.py               # Temporal Graph Isomorphism Network
│       ├── action_scorer.py      # Hybrid scorer: linear (frozen) + MLP residual
│       └── graph_builder.py      # Builds HeteroData from observations
│
├── rewards/
│   ├── reward_fn.py              # Main reward dispatcher
│   ├── reward_weights.yaml       # All weights (v5 final)
│   └── components/
│       ├── maintenance_reward.py # r₁ computation
│       ├── scheduling_reward.py  # r₂ computation
│       └── shared_reward.py      # R_shared computation
│
├── training/
│   ├── mappo_trainer.py          # Main training loop (mask storage, v2=r2_mean)
│   ├── ppo_update.py             # PPO clipped objective with stored masks
│   └── rollout_buffer.py         # Experience buffer with action mask support
│
├── utils/
│   ├── checkpoint.py             # Save/load (includes ActionScorer weights)
│   └── logger.py                 # TensorBoard logging
│
├── benchmarks/
│   └── baselines.py              # 4 heuristic baselines (Reactive, Fixed-PM, ABR, Rule-EDF)
│
├── analytics/
│   ├── episode_kpis.py           # 24 conference-grade KPIs
│   ├── excel_writer.py           # Excel output utilities
│   └── plot_utils.py             # Plotting utilities
│
├── scripts/
│   ├── train.py                  # Training entry point (linear head freeze on resume)
│   ├── eval_checkpoint.py        # MARL vs baselines evaluation
│   ├── run_all_ablations.py      # Full ablation suite (25+ configs)
│   ├── statistical_inference.py  # Pairwise tests, Pareto, win rates
│   ├── bc_warmstart.py           # Behavioural cloning experiment
│   ├── training_impact_analysis.py # Pre-BC vs post-BC comparison
│   ├── analyze_training.py       # TensorBoard → curves + early/late analysis
│   ├── extract_tensorboard.py    # TB → CSV/Excel export
│   ├── identify_runs.py          # Identify TB run folders
│   └── regenerate_plots.py       # Regenerate plots from saved CSVs
│
├── tests/                        # 55 unit tests
├── run_baselines.py              # Standalone baseline benchmark suite
├── analyze_resources.py          # Inventory sawtooth analysis
├── compare_results.py            # Standalone statistical comparison
├── test_resources_exhaustive.py  # 28 resource dynamics tests
│
├── thesis.tex                    # Full LaTeX thesis (1287 lines)
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

---

## Key Design Decisions

### 1. Health Gate (h < 75%)

Without this constraint, any positive PM reward signal causes 120+ PMs/episode (31% of capacity wasted). The hard gate is enforced in both execution (`_step_agent1`) and action masking (`action_spaces.py`). Baselines run with `bypass_health_gate=True` to use their own PM timing.

### 2. Expert-Initialised Frozen Linear Head

RL cannot improve on the expert initialisation in this setting. Within 200 RL episodes, gradient updates corrupt the expert weights and jobs drop from 24 → 18. The frozen linear head guarantees a performance floor while the MLP residual learns corrections (urgency tie-breaking, health-aware routing).

### 3. Running-Mean Baseline for Agent 2

Using the centralised critic (trained on r₁) for Agent 2 advantages produced noise — the critic predicts maintenance value, which is uninformative for scheduling. The running-mean baseline `v₂ = EMA(r₂)` is simpler but correct: advantages reflect per-step scheduling quality.

### 4. Action Mask in PPO Update

The action mask must be stored during rollout collection and reapplied during the PPO update. Without stored masks, importance ratios are corrupted (collection uses masked distribution, update uses unmasked). This caused Agent 1's entropy to remain stuck at 3.09 for 120k steps.

### 5. Bypass Health Gate for Baselines

Baselines like Fixed-PM (PM every 30 shifts, when health=97%) are blocked by the h<75% gate designed for training. The `bypass_health_gate` flag (default `False`, set `True` for evaluation) lets baselines use their own PM timing.

---

## Reproducing Results

```bash
# 1. Train Phase 1 from scratch
python scripts/train.py --config configs/phase1.yaml --timesteps 225000

# 2. Evaluate trained checkpoint
python scripts/eval_checkpoint.py \
    --checkpoint outputs/checkpoints/latest.pt \
    --seed-sets 3 --episodes 30

# 3. Run ablation study
python scripts/run_all_ablations.py \
    --checkpoint outputs/checkpoints/latest.pt

# 4. Statistical analysis
python scripts/statistical_inference.py --datadir outputs/ablation_study

# 5. Training analysis
python scripts/training_impact_analysis.py
```

**Hardware used:** NVIDIA GPU with CUDA 12.1, ~2 hours for 225k steps. CPU training possible but ~10x slower.

**Expected output:** Results within ±1 standard deviation of reported values. Exact reproduction requires fixing all random seeds (environment, PyTorch, numpy).

---

## Citation

```bibtex
@thesis{jha2026marl_manufacturing,
  title   = {Multi-Agent Reinforcement Learning for Joint Predictive 
             Maintenance and Flexible Job Shop Scheduling in 
             Stochastic Manufacturing Environments},
  author  = {Jha, Shreenath and Gehlot, Kirtan},
  year    = {2026},
  school  = {Indian Institute of Technology Delhi},
  type    = {B.Tech Project (MCD412)},
  advisor = {Kumari, Minakshi}
}
```

## Acknowledgements

We thank Dr. Minakshi Kumari for supervision and guidance throughout this project.

## License

Academic use only. IIT Delhi B.Tech Project.