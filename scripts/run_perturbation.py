#!/usr/bin/env python3
"""Run perturbation analysis (Section 4.3)."""
import sys
sys.path.insert(0, "..")


def perturb_edge(graph_state, perturbation_type):
    """Apply single-edge perturbation and measure embedding change."""
    pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--instance-dir", default="../data/instances/15x10")
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--output", default="../experiments/perturbation/results.csv")
    args = parser.parse_args()
    print(f"Perturbation analysis | trials={args.n_trials}")


if __name__ == "__main__":
    main()
