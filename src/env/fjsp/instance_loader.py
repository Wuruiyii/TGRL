import json
import os


def load_instance(filepath: str) -> dict:
    """Load FJSP-AGV instance from JSON."""
    with open(filepath) as f:
        data = json.load(f)
    return data


def load_all_instances(directory: str, scale: str = None) -> list[dict]:
    instances = []
    for fn in sorted(os.listdir(directory)):
        if fn.endswith(".json"):
            if scale and scale not in fn:
                continue
            instances.append(load_instance(os.path.join(directory, fn)))
    return instances
