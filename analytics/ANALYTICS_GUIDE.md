# Analytics Guide — BTP2 MARL Manufacturing
## What Changed, When to Run What, Where Everything Goes

---

## PART 0 — WHAT CHANGED FROM YOUR OLD FILES

### New folder: `analytics/` (create in project root)
This didn't exist before. It's the shared library that all scripts now import from.

```
analytics/__init__.py         empty
analytics/episode_kpis.py     28 KPIs computed per episode (was 8 in old code)
analytics/excel_writer.py     dark-background Excel workbooks (new)
analytics/plot_utils.py       dark matplotlib figures (new)
```

### `analyze_baselines.py` — complete rewrite
**Old:** 8 KPIs (failures, n_PM, n_CM, tardiness, completions, service_level, avg_health, pm_cm_ratio). Single bar chart. Basic Excel.

**New:**
- 28 KPIs across Reliability, Scheduling, Cost/Inventory domains
- All 3 stochasticity levels run separately (stoch=1,2,3)
- Exports: `comparison_table.xlsx` (Summary + Episodes + Statistics sheets), `bar_comparison.png`, `radar_chart.png`, `episode_data.csv`
- CSV is the input to `compare_results.py`

### `analyze_training.py` — complete rewrite
**Old:** Basic training curves, minimal early/late comparison.

**New:**
- `training_curves.png` — 8-panel with target lines (e.g. availability target 0.85)
- `resource_dynamics.png` — renewable + consumable over training (detects Bug 4/5 visually)
- `reward_decomposition.png` — bar chart showing signal magnitudes
- `convergence_report.txt` — steps to availability>0.8, entropy decay, critic loss check
- `training_analysis.xlsx` — early vs late table + full TB summary

### `compare_results.py` — complete rewrite
**Old:** t-test only, basic Cohen's d.

**New:**
- Wilcoxon signed-rank (paired, non-parametric — correct for RL)
- Holm-Bonferroni correction for multiple comparisons
- Cohen's d with 95% bootstrap CI (2000 resamples)
- Cliff's delta (ordinal effect size)
- Excel output includes a Notes sheet explaining how to cite each result
- Prints p-value significance stars (*** p<0.001, ** p<0.01, * p<0.05)

### `analyze_checkpoints.py` — complete rewrite
**Old:** minimal.

**New:**
- Evaluates each `phase*_step_*k.pt` checkpoint on N episodes
- Shows KPI evolution: availability, failures, service_level, total_cost, MTBF, RUL per checkpoint
- Plots against ABR baseline reference line
- Exports `learning_curve.png` and `checkpoint_kpis.xlsx`

### `Ablations/check_convergence.py` — complete rewrite
**Old:** checked one metric.

**New:** Three GO criteria (all must pass):
1. Mean availability (last window episodes) >= 0.80
2. Entropy agent1 (last 10 updates) < 3.0 (declining from 3.47)
3. Critic loss (last 10 updates) > 0 (actually training)

Plus informational: failures/ep, CM events appeared, r_shared non-zero.

### `global_analytics.py` — NEW (didn't exist before)
Master orchestrator. Runs everything in sequence: baselines × 3 stoch levels, training analysis, ablations, statistics, and builds `MASTER_TABLE.xlsx`.

---

## PART 1 — FULL OUTPUT FOLDER STRUCTURE

Everything goes under `outputs/`. Never in the project root.

```
outputs/
├── checkpoints/
│   ├── latest.pt                    overwritten every 50k steps
│   ├── phase1_step_050k.pt          permanent snapshot
│   ├── phase1_step_100k.pt
│   ├── phase1_end.pt                MANUAL: save after phase converges
│   ├── phase2_step_150k.pt
│   ├── phase2_end.pt
│   ├── phase3_step_200k.pt
│   └── phase3_end.pt                FINAL checkpoint
│
├── runs/
│   ├── phase1_1775500000/           TensorBoard events (auto-named)
│   ├── phase2_1775510000/
│   └── phase3_1775520000/
│
├── logs/
│   └── training_log.txt             written by train_overnight.py
│
└── results/
    ├── baselines/
    │   ├── stoch1/                  Phase 1 comparison (deterministic)
    │   │   ├── comparison_table.xlsx
    │   │   ├── bar_comparison.png
    │   │   ├── radar_chart.png
    │   │   └── episode_data.csv     INPUT for compare_results.py
    │   ├── stoch2/                  Phase 2 comparison (+noise)
    │   └── stoch3/                  Phase 3 comparison (main paper result)
    │
    ├── training/
    │   ├── training_curves.png      8-panel: avail, fail, critic_loss, entropy, r1, r2, shared, PM
    │   ├── resource_dynamics.png    renewable + consumable over training
    │   ├── reward_decomposition.png signal magnitude bar chart
    │   ├── convergence_report.txt   steps to avail>0.8, entropy decay
    │   └── training_analysis.xlsx   early vs late + full TB summary
    │
    ├── checkpoints/
    │   ├── learning_curve.png       KPI vs training step per checkpoint
    │   └── checkpoint_kpis.xlsx     table: all KPIs per checkpoint
    │
    ├── ablations/
    │   ├── A9_pm_timing/
    │   │   ├── pm_age_histogram.png  FIGURE 4 in paper
    │   │   └── pm_timing_results.xlsx
    │   ├── A1_delta_rul/
    │   ├── A8_zeroshot/
    │   ├── A10_health_dispatch/
    │   ├── A4_indep_ppo/
    │   └── A12_lambda/
    │
    ├── stats/
    │   └── significance_table.xlsx  Wilcoxon + Cohen's d + Holm-Bonferroni
    │
    └── paper/                       global_analytics.py destination
        └── MASTER_TABLE.xlsx        all paper tables in one workbook
```

---

## PART 2 — WHEN TO RUN WHAT

### Before ANY training (one-time setup)
```bash
python -m pytest tests/ -q
# Expected: 91 passed

python test_resources_exhaustive.py
# Expected: 27/27 passed

python verify_fix.py
# Expected: ALL LIVE CHECKS PASSED (takes ~30 seconds)

python time_check.py
# Tells you: Phase1 ~5h, Phase2 ~2h, Phase3 ~2h on RTX 3060
```

---

### During Phase 1 training — every 30k steps

**Start training and TensorBoard in separate terminals:**

Terminal 1 — Training:
```bash
python train_overnight.py
```

Terminal 2 — TensorBoard (open http://localhost:6006):
```bash
tensorboard --logdir outputs/runs/
```

Terminal 3 — Convergence check every 30k steps (or when train_overnight.py pauses):
```bash
python Ablations/check_convergence.py --window 50
```

**What to look for in TensorBoard:**
- `train/critic_loss` — must be non-zero from step 0
- `episode/n_CM` — must appear by episode 50
- `episode/availability` — must be rising
- `debug/renewable_0` — should not get stuck at 0 (Bug 4 indicator)
- `rewards/shared` — must be non-zero on failure steps

**GO criteria** (check_convergence.py output must say GO):
- Availability mean (last 50 eps) >= 0.80
- Entropy agent1 < 3.0 (declining from 3.47)
- Critic loss > 0

**If GO — save Phase 1 end checkpoint:**
```bash
# Windows:
copy outputs\checkpoints\latest.pt outputs\checkpoints\phase1_end.pt

# Linux/Mac:
cp outputs/checkpoints/latest.pt outputs/checkpoints/phase1_end.pt
```

**Run Phase 1 analytics:**
```bash
python analyze_training.py --logdir outputs/runs/ --outdir outputs/results/training/

python analyze_baselines.py \
    --checkpoint outputs/checkpoints/phase1_end.pt \
    --episodes 50 \
    --stoch-level 1 \
    --outdir outputs/results/baselines/stoch1/

python analyze_checkpoints.py \
    --ckpt-dir outputs/checkpoints/ \
    --episodes 20 \
    --outdir outputs/results/checkpoints/
```

---

### Phase 2 training

```bash
python scripts/train.py \
    --config configs/phase2.yaml \
    --timesteps 50000 \
    --resume outputs/checkpoints/phase1_end.pt
```

```bash
# Monitor convergence
python Ablations/check_convergence.py --window 30 --avail-target 0.82

# Save when done
copy outputs\checkpoints\latest.pt outputs\checkpoints\phase2_end.pt
```

**Run Phase 2 baseline comparison:**
```bash
python analyze_baselines.py \
    --checkpoint outputs/checkpoints/phase2_end.pt \
    --episodes 50 \
    --stoch-level 2 \
    --outdir outputs/results/baselines/stoch2/
```

---

### Phase 3 training

```bash
python scripts/train.py \
    --config configs/phase3.yaml \
    --timesteps 50000 \
    --resume outputs/checkpoints/phase2_end.pt
```

```bash
copy outputs\checkpoints\latest.pt outputs\checkpoints\phase3_end.pt
```

---

### After Phase 3 — full analysis run

Run these in order. Each one feeds into the next.

**Step 1: Phase 3 baseline comparison (main paper Table 1)**
```bash
python analyze_baselines.py \
    --checkpoint outputs/checkpoints/phase3_end.pt \
    --episodes 100 \
    --stoch-level 3 \
    --outdir outputs/results/baselines/stoch3/
```
Output: `outputs/results/baselines/stoch3/comparison_table.xlsx`, `bar_comparison.png`, `radar_chart.png`, `episode_data.csv`

**Step 2: Statistical significance (paper Table 3)**
```bash
python compare_results.py \
    --csv outputs/results/baselines/stoch3/episode_data.csv \
    --marl "MARL (ours)" \
    --baseline "ABR+MDD+(Q,R)" \
    --outdir outputs/results/stats/
```
Output: `outputs/results/stats/significance_table.xlsx` — Wilcoxon p-values, Cohen's d, effect sizes

**Step 3: Training curve analysis (paper Figure 2)**
```bash
python analyze_training.py \
    --logdir outputs/runs/ \
    --outdir outputs/results/training/
```
Output: `training_curves.png`, `resource_dynamics.png`, `convergence_report.txt`, `training_analysis.xlsx`

**Step 4: Learning curve across checkpoints (paper Figure 3)**
```bash
python analyze_checkpoints.py \
    --ckpt-dir outputs/checkpoints/ \
    --episodes 30 \
    --outdir outputs/results/checkpoints/
```
Output: `learning_curve.png`, `checkpoint_kpis.xlsx`

**Step 5: Ablation studies Group 1 (~90 min) (paper Figure 4, Table 4)**
```bash
# A9: PM timing vs ABR optimal t* (Figure 4)
python Ablations/A9_pm_timing.py \
    --checkpoint outputs/checkpoints/phase3_end.pt \
    --episodes 100 \
    --outdir outputs/results/ablations/A9_pm_timing/

# A1: ΔRUL signal ablation — does health-based reward help? (answers RQ2)
python Ablations/A1_delta_rul.py \
    --checkpoint outputs/checkpoints/phase3_end.pt \
    --episodes 100 \
    --outdir outputs/results/ablations/A1_delta_rul/

# A10: Health-aware dispatch — does Agent 2 route around degraded machines?
python Ablations/A10_health_dispatch.py \
    --checkpoint outputs/checkpoints/phase3_end.pt \
    --episodes 100 \
    --outdir outputs/results/ablations/A10_health_dispatch/
```

**Step 6: Ablation studies Group 2 (~2 hours)**
```bash
# A8: Zero-shot M=5 → M=10 scaling (TGIN generalisation)
python Ablations/A8_zeroshot_scaling.py \
    --checkpoint outputs/checkpoints/phase3_end.pt \
    --outdir outputs/results/ablations/A8_zeroshot/

# A4: MARL vs independent PPO (coordination benefit)
python Ablations/A4_independent_ppo.py \
    --outdir outputs/results/ablations/A4_indep_ppo/

# A12: λ sensitivity (what coupling coefficient is optimal?)
python Ablations/A12_lambda_sensitivity.py \
    --checkpoint outputs/checkpoints/phase3_end.pt \
    --outdir outputs/results/ablations/A12_lambda/
```

**Step 7: Master table for paper (all results in one workbook)**
```bash
python global_analytics.py \
    --phase3-ckpt outputs/checkpoints/phase3_end.pt \
    --episodes 100 \
    --outdir outputs/results/paper/ \
    --skip-ablations
```
Output: `outputs/results/paper/MASTER_TABLE.xlsx` aggregates all comparison tables

**Step 8: Run EVERYTHING at once (alternative to steps 1-7)**
```bash
python global_analytics.py \
    --phase3-ckpt outputs/checkpoints/phase3_end.pt \
    --episodes 100 \
    --outdir outputs/results/paper/
```

---

## PART 3 — PAPER TABLE / FIGURE MAPPING

| Paper element | Script | Output file |
|---|---|---|
| Table 1: MARL vs baselines | `analyze_baselines.py --stoch-level 3` | `baselines/stoch3/comparison_table.xlsx` |
| Table 2: Statistical significance | `compare_results.py` | `stats/significance_table.xlsx` |
| Table 3: KPI summary | `analyze_checkpoints.py` | `checkpoints/checkpoint_kpis.xlsx` |
| Figure 1: Training curves | `analyze_training.py` | `training/training_curves.png` |
| Figure 2: Learning curve | `analyze_checkpoints.py` | `checkpoints/learning_curve.png` |
| Figure 3: PM timing | `A9_pm_timing.py` | `ablations/A9_pm_timing/pm_age_histogram.png` |
| Figure 4: Radar chart | `analyze_baselines.py` | `baselines/stoch3/radar_chart.png` |
| Appendix: Resource dynamics | `analyze_training.py` | `training/resource_dynamics.png` |
| Appendix: Phase 1 comparison | `analyze_baselines.py --stoch-level 1` | `baselines/stoch1/comparison_table.xlsx` |

---

## PART 4 — QUICK REFERENCE

### Most-used commands

```bash
# Check convergence (during training)
python Ablations/check_convergence.py --window 50

# Full baseline comparison
python analyze_baselines.py --checkpoint outputs/checkpoints/phase3_end.pt --episodes 100 --stoch-level 3 --outdir outputs/results/baselines/stoch3/

# Statistical tests
python compare_results.py --csv outputs/results/baselines/stoch3/episode_data.csv --outdir outputs/results/stats/

# Training analysis
python analyze_training.py --outdir outputs/results/training/

# Learning curve
python analyze_checkpoints.py --ckpt-dir outputs/checkpoints/ --episodes 30 --outdir outputs/results/checkpoints/

# Everything at once
python global_analytics.py --phase3-ckpt outputs/checkpoints/phase3_end.pt --episodes 100 --outdir outputs/results/paper/
```

### Interpreting statistical output (compare_results.py)

```
  Metric          MARL mean  ABR mean    Δ%     p        HB   d      Effect       Win?
  availability    0.8712     0.7541   +15.5%  0.0001***  ✓  1.823  large        ✓

p:   *** p<0.001  ** p<0.01  * p<0.05  (raw)
HB:  ✓ = also significant after Holm-Bonferroni correction (conservative)
d:   Cohen's d — negligible<0.2, small<0.5, medium<0.8, large≥0.8
Win: ✓ = MARL is statistically better  ✗ = MARL is worse  ? = not significant
```

**Cite in paper as:**
> "MARL achieves availability 0.87±0.04 vs ABR 0.75±0.06 (Wilcoxon p<0.001, d=1.8, large effect)"

---

## PART 5 — FILE PLACEMENT SUMMARY

```
# New folder to create:
analytics/
  __init__.py
  episode_kpis.py
  excel_writer.py
  plot_utils.py

# Files to REPLACE (overwrite existing):
analyze_baselines.py
analyze_training.py
analyze_checkpoints.py
compare_results.py
global_analytics.py
Ablations/check_convergence.py
```

All files downloaded from the analytics_suite folder in your downloads.
