#!/usr/bin/env python3
"""Benchmark inference latency (Section 6.6)."""
import sys
sys.path.insert(0, "..")
from src.utils.timer import InferenceTimer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scale", default="50x20")
    parser.add_argument("--n-warmup", type=int, default=50)
    parser.add_argument("--n-measure", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(f"Latency benchmark | scale={args.scale} | device={args.device}")


if __name__ == "__main__":
    main()
