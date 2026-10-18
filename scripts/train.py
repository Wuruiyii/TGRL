#!/usr/bin/env python3
"""TGRL training entry point."""
import argparse
import yaml
import sys
sys.path.insert(0, "..")
from src.models.ppo import PPOAgent
from src.env import FJSPAGVEnv, load_instance
from src.utils.curriculum import CurriculumScheduler
from src.utils.seed_manager import set_global_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/configs/default.yaml")
    parser.add_argument("--instance-dir", default="data/instances/15x10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_global_seed(args.seed)
    scheduler = CurriculumScheduler(
        pool_capacity=config["curriculum"]["pool_capacity"],
        beta=config["curriculum"]["priority_beta"],
    )
    agent = PPOAgent(config["model"])

    print(f"Training TGRL | config={args.config} | seed={args.seed}")
    print(f"Curriculum stages: {config['curriculum']['stages']}")
    print(f"Total episodes: {config['training']['total_episodes']}")
    # ... training loop omitted for brevity


if __name__ == "__main__":
    main()
