import numpy as np
import torch
import json
import os


def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_seeds(path: str) -> list[int]:
    with open(path) as f:
        return json.load(f)


def generate_seed_file(path: str, n: int = 30):
    seeds = np.random.RandomState(42).randint(0, 2**31, size=n).tolist()
    with open(path, "w") as f:
        json.dump(seeds, f)
    return seeds
