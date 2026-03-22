"""
scripts/evaluate.py
====================
Evaluates trained policy on benchmark instances.

Usage:
    python scripts/evaluate.py --config configs/benchmark_instances/small_3m_5j.yaml
    python scripts/evaluate.py --config configs/benchmark_instances/medium_5m_10j.yaml
                                --checkpoint checkpoints/latest.pt --episodes 20
"""

import argparse
import yaml
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/base.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes",   type=int, default=10)
    parser.add_argument("--render",     action="store_true")
    parser.add_argument("--seed",       type=int, default=999)
    args = parser.parse_args()

    base_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "configs", "base.yaml")
    with open(base_path) as f:
        config = yaml.safe_load(f)

    if args.config != base_path:
        with open(args.config) as f:
            override = yaml.safe_load(f)
        if override:
            config.update(override)

    from benchmarks.evaluate import evaluate_policy
    evaluate_policy(
        config=config,
        ckpt_path=args.checkpoint,
        n_episodes=args.episodes,
        render=args.render,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()