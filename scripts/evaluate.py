#!/usr/bin/env python3
"""Evaluate trained TGRL agent on benchmark instances."""
import argparse
import json
import csv
import sys
sys.path.insert(0, "..")
from src.env import FJSPAGVEnv, load_instance
from src.utils.seed_manager import load_seeds
from src.utils.timer import InferenceTimer
from src.utils.metrics import normalised_makespan, mean_tardiness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--instance-dir", required=True)
    parser.add_argument("--seeds", default="data/seeds/replication_seeds.json")
    parser.add_argument("--profile", default="D4")
    parser.add_argument("--output", default="results.csv")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seeds = load_seeds(args.seeds)
    print(f"Evaluating | profile={args.profile} | replications={len(seeds)}")


if __name__ == "__main__":
    main()
