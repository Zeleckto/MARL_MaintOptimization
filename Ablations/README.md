# Ablation Studies — BTP2 MARL Manufacturing

Complete ablation study package for the paper. Each script is self-contained
and produces JSON + table outputs ready for the paper.

## Quick Start

```bash
# After Phase 3 training completes:
cd D:\IITD\SEM8\BTP2\Code\v0\manufacturing_marl

# Run essential ablations (Group 1, ~2h)
python ablations/run_all_ablations.py \
    --checkpoint checkpoints/phase3_500k.pt \
    --group 1 \
    --outdir results/ablations/

# Check if current training has converged (run any time)
python ablations/check_convergence.py --outdir results/convergence/
```

---

## Files

| File | Ablation | RQ Answered | Est. Time |
|---|---|---|---|
| `ablation_utils.py` | Shared utilities | — | — |
| `check_convergence.py` | GO/NO-GO monitor | — | 1 min |
| `A1_delta_rul.py` | ΔRUL signal on/off | **RQ2** | 30 min |
| `A2_fail_idle_sensitivity.py` | w_fail_idle: 0.5/1/2/4 | Design | 45 min |
| `A4_independent_ppo.py` | IndepPPO vs MAPPO | RQ1 | 15 min |
| `A8_zeroshot_scaling.py` | M=5 → M=10 zero-shot | Scale | 20 min |
| `A9_pm_timing.py` | PM age distribution vs ABR t* | RQ1 | 30 min |
| `A10_health_dispatch.py` | Health-conditioned assignment | RQ1 | 30 min |
| `A12_lambda_sensitivity.py` | λ ∈ {0, 0.1, 0.3, 0.6, 1.0} | Design | 45 min |
| `run_all_ablations.py` | Orchestrator | — | — |

---

## Ablation Details

### A1: ΔRUL Signal (MUST DO — answers RQ2)

**What:** Evaluate trained checkpoint with w_RUL=0.05 vs w_RUL=0.0

**Expected:** PM events drop by 30–50% without ΔRUL. Failures rise.

**Paper claim:** "The ΔRUL formulation provides immediate PM credit,
resolving the γ^40=0.67 discounting problem. Without ΔRUL, PM frequency
drops by X% (p<0.05), confirming its necessity."

```bash
python ablations/A1_delta_rul.py \
    --checkpoint checkpoints/phase3_500k.pt \
    --episodes 100 \
    --outdir results/ablation_a1/
```

---

### A2: w_fail_idle Sensitivity

**What:** Test the idle-FAIL penalty at 0.5, 1.0, 2.0, 4.0

**Expected:** Too low (0.5) = CM events rare. Too high (4.0) = r1 noisy.
Current 2.0 = CM emerges within 4 FAIL-steps (break-even analysis).

```bash
python ablations/A2_fail_idle_sensitivity.py \
    --checkpoint checkpoints/phase3_500k.pt \
    --episodes 50
```

---

### A4: Independent PPO vs MAPPO (Retrains on Small Instance)

**What:** Train M=3, J=10 for 20k steps under two conditions:
- MAPPO: centralised critic + λ=0.3
- IndepPPO: separate critics + λ=0.0

**Why small instance:** Gets meaningful results in ~15 minutes.
Full M=5 retraining would take 15+ hours.

**Expected:** MAPPO > IndepPPO on failures/ep and joint metrics.

```bash
python ablations/A4_independent_ppo.py \
    --timesteps 20000 \
    --eval-episodes 50 \
    --outdir results/ablation_a4/
```

---

### A8: Zero-Shot Scaling M=5 → M=10 (FREE — no retraining)

**What:** Load M=5 trained checkpoint, evaluate on M=10 environment.
Agent 2 (TGIN): true zero-shot. Agent 1 (MLP): first 5 machines only.

**Expected:** Within 20–30% of M=5 performance (TGIN permutation invariance).

```bash
python ablations/A8_zeroshot_scaling.py \
    --checkpoint checkpoints/phase3_500k.pt \
    --episodes 20
```

---

### A9: PM Timing Distribution (MUST DO — strongest visual)

**What:** Log machine age at every PM initiation across 50 episodes.
Compare histogram vs ABR optimal t* (49–65 shifts).

**If histogram peaks near ABR t*:** MARL independently rediscovered the
analytical optimum — strongest single result in the paper.

```bash
python ablations/A9_pm_timing.py \
    --checkpoint checkpoints/phase3_500k.pt \
    --episodes 50 \
    --outdir results/ablation_a9/
```

---

### A10: Health-Conditioned Dispatch

**What:** At every Agent 2 assignment, compare health of chosen machine vs
healthiest available alternative.

**If health gap ≥ 0 significantly:** Agent 2 has learned to prefer
healthier machines — the emergent cooperative behaviour.

```bash
python ablations/A10_health_dispatch.py \
    --checkpoint checkpoints/phase3_500k.pt \
    --episodes 50
```

---

### A12: λ Coupling Sensitivity

**What:** Test λ ∈ {0, 0.1, 0.3, 0.6, 1.0} on trained checkpoint.

**λ=0 vs λ=0.3:** Tests whether cooperative failure signal is needed.

```bash
python ablations/A12_lambda_sensitivity.py \
    --checkpoint checkpoints/phase3_500k.pt \
    --episodes 50
```

---

## Convergence Checker

Run any time during or after training:

```bash
python ablations/check_convergence.py
# GO criterion: all 5 mandatory checks pass
# Mandatory: n_CM>0, n_PM>0, failures↓, entropy>1.0, availability>0.90
```

---

## Priority Order for BTP2 Deadline

If time is tight, run in this order:

1. `check_convergence.py` — 1 min, tells you if training is done
2. `A9_pm_timing.py` — 30 min, **strongest visual for paper**
3. `A1_delta_rul.py` — 30 min, **answers RQ2 directly**
4. `A10_health_dispatch.py` — 30 min, shows cooperative behaviour
5. `A2_fail_idle_sensitivity.py` — 45 min, Appendix B material
6. `A12_lambda_sensitivity.py` — 45 min, Appendix B material
7. `A8_zeroshot_scaling.py` — 20 min, free generalisation result
8. `A4_independent_ppo.py` — 15 min, proves MARL benefit

Total Group 1 (essential): ~90 min
Total all ablations: ~3.5h
