"""
scripts/train.py
================
Entry point for training.

Usage:
    python scripts/train.py --config configs/phase1.yaml
    python scripts/train.py --config configs/phase2.yaml --timesteps 1000000
    python scripts/train.py --config configs/phase1.yaml --resume checkpoints/latest.pt
"""

import argparse
import os
import sys

# Auto-clear __init__.py files before any imports
# Prevents circular import crashes caused by auto-imports in package inits
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _dirpath, _dirs, _files in os.walk(_root):
    if "venv" in _dirpath or ".git" in _dirpath:
        continue
    for _f in _files:
        if _f == "__init__.py":
            _p = os.path.join(_dirpath, _f)
            try:
                if os.path.getsize(_p) > 0:
                    open(_p, "w").close()
            except Exception:
                pass

sys.path.insert(0, _root)
import yaml


def load_config(path: str) -> dict:
    """Loads phase config on top of base config."""
    base_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "base.yaml"
    )
    with open(base_path) as f:
        config = yaml.safe_load(f)

    if path != base_path:
        with open(path) as f:
            override = yaml.safe_load(f)
        if override:
            config.update(override)

    return config


def main():
    parser = argparse.ArgumentParser(description="Train MAPPO manufacturing agent")
    parser.add_argument("--config",     default="configs/phase1.yaml")
    parser.add_argument("--timesteps",  type=int, default=500_000)
    parser.add_argument("--resume",     default=None, help="checkpoint path to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Config loaded: stochasticity_level={config['stochasticity_level']}")

    from training.mappo_trainer import MAPPOTrainer
    trainer = MAPPOTrainer(config)

    if args.resume:
        print(f"Resuming from: {args.resume}")
        # TODO: load checkpoint into trainer

    trainer.train(total_timesteps=args.timesteps)


if __name__ == "__main__":
    main()