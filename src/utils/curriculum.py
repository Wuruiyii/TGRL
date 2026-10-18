"""Four-stage curriculum with priority-aware scenario sampling (Section 5.4)."""
import numpy as np


STAGE_PROFILES = ["D0", "D1", "D2", "D4"]
EPISODES_PER_STAGE = 5000


class CurriculumScheduler:
    def __init__(self, pool_capacity=2000, beta=0.6, eps_p=0.01):
        self.pool_capacity = pool_capacity
        self.beta = beta
        self.eps_p = eps_p
        self._pool = []
        self._stage = 0
        self._episode = 0

    @property
    def current_profile(self):
        return STAGE_PROFILES[min(self._stage, len(STAGE_PROFILES) - 1)]

    @property
    def stage(self):
        return self._stage

    def step(self):
        self._episode += 1
        if self._episode % EPISODES_PER_STAGE == 0 and self._stage < len(STAGE_PROFILES) - 1:
            self._stage += 1
            self._pool.clear()

    def add_scenario(self, descriptor):
        self._pool.append(descriptor)
        if len(self._pool) > self.pool_capacity:
            self._pool.pop(0)

    def sample(self, n=32, rng=None):
        if not self._pool:
            return []
        rng = rng or np.random.RandomState()
        weights = np.array([self._priority_weight(s) for s in self._pool])
        weights /= weights.sum()
        indices = rng.choice(len(self._pool), size=min(n, len(self._pool)),
                             replace=False, p=weights)
        return [self._pool[i] for i in indices]

    def _priority_weight(self, scenario):
        nd = scenario.get("n_disruptions", 0)
        n_bar = np.mean([s.get("n_disruptions", 0) for s in self._pool]) if self._pool else 1.0
        return (nd / max(n_bar, 1e-6) + self.eps_p) ** self.beta
