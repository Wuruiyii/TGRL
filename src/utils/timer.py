import time
import torch


class InferenceTimer:
    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self._records = []

    def start(self):
        if self.use_cuda:
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def stop(self):
        if self.use_cuda:
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - self._t0) * 1000
        self._records.append(elapsed)
        return elapsed

    @property
    def median_ms(self):
        import numpy as np
        return float(np.median(self._records)) if self._records else 0.0

    @property
    def mean_ms(self):
        import numpy as np
        return float(np.mean(self._records)) if self._records else 0.0

    def reset(self):
        self._records.clear()
