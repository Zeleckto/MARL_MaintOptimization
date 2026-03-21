# Three-Tier Manufacturing Optimization — MARL

**BTP2 Project | IIT Delhi | Dept. of Mechanical Engineering**

Students: Shreenath Jha (2022ME11306) · Kirtan Gehlot (2022ME12030)  
Supervisor: Dr. Minakshi Kumari

---

## What This Is

A Multi-Agent Reinforcement Learning framework for integrated manufacturing optimization:
- **Predictive Maintenance** (Agent 1 — MLP policy)
- **Job Shop Scheduling** (Agent 2 — Tripartite Graph Isomorphism Network)
- **Resource Allocation** (managed by Agent 1)

Three-tier architecture:
- **Tier 1** — OR-Tools CP-SAT exact solver (optimal benchmarks for small instances)
- **Tier 2** — Offline learning / imitation (planned future work)
- **Tier 3** — MAPPO online RL (main contribution)

---

## Setup

```bash
# 1. Clone
git clone https://github.com/Zeleckto/MARL_MaintOptimization.git
cd MARL_MaintOptimization

# 2. Create virtual environment
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash
# source venv/bin/activate         # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install PyG (match your CUDA version — check: nvcc --version)
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
# Replace cu121 with cpu if no GPU
```

---

## Project Structure

```
manufacturing_marl/
├── configs/                # Phase configs (phase1/2/3.yaml) + benchmark instances
├── environments/           # PettingZoo AEC environment
│   └── transitions/        # Physics: degradation, jobs, resources, failures
├── models/
│   └── tgin/               # Tripartite GIN for Agent 2
├── agents/                 # PDM agent (Agent 1) + Job Shop agent (Agent 2)
├── training/               # MAPPO trainer, rollout buffer, PPO update
├── rewards/                # Decomposed reward components
├── tier1/                  # OR-Tools CP-SAT formulation (38 constraints)
├── benchmarks/             # Tier 1 vs Tier 3 evaluation
├── utils/                  # Seeding, logging, distributions, checkpointing
└── tests/                  # Unit tests per module
```

---

## Running

```bash
# Train (Phase 1)
python scripts/train.py --config configs/phase1.yaml

# Render one episode
python scripts/render_episode.py --config configs/phase1.yaml --checkpoint checkpoints/latest.pt

# Evaluate vs Tier 1
python scripts/evaluate.py --config configs/benchmark_instances/small_3m_5j.yaml
```

---

## Current Status

| Component | Status |
|---|---|
| `utils/` — seeding, distributions | ✅ Complete + tested |
| `environments/transitions/degradation.py` | ✅ Complete + tested |
| `configs/base.yaml` | ✅ Complete |
| Everything else | 🔧 In progress |

---

## Key Design Decisions

- **PettingZoo AEC** (not ParallelEnv) — enforces Agent1→Agent2 action ordering, eliminates concurrent action conflicts
- **PyTorch Geometric HeteroConv** for TGIN — tripartite graph: Operations × Machines × Jobs
- **Separated reward components** — r1 (PDM), r2 (scheduling), R_shared (failures) for clean credit assignment
- **Weibull + Kijima Type I** physics — effective_age = virtual_age + time_since_maint
- **3 stochasticity phases** — train on Phase 1 first, validate learning before adding noise
