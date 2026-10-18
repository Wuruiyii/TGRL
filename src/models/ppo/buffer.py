import torch
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Transition:
    state: dict
    action: dict
    reward: float
    done: bool
    log_prob: float
    value: float
    hidden: object = None


class RolloutBuffer:
    def __init__(self, capacity: int = 8192):
        self._capacity = capacity
        self._buffer: list[Transition] = []

    def push(self, transition: Transition):
        self._buffer.append(transition)
        if len(self._buffer) > self._capacity:
            self._buffer.pop(0)

    def sample_minibatch(self, batch_size: int = 256):
        indices = np.random.choice(len(self._buffer), batch_size, replace=False)
        return [self._buffer[i] for i in indices]

    def clear(self):
        self._buffer.clear()

    def __len__(self):
        return len(self._buffer)

    @property
    def is_full(self):
        return len(self._buffer) >= self._capacity
