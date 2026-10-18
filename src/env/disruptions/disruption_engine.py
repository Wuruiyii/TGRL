import numpy as np

PROFILES = {
    "D0": {"sigma_p": 0.0, "lambda_b": 0.0, "lambda_u": 0.0, "lambda_v": 0.0, "congestion": False},
    "D1": {"sigma_p": 0.05, "lambda_b": 0.0, "lambda_u": 0.0, "lambda_v": 0.0, "congestion": False},
    "D2": {"sigma_p": 0.10, "lambda_b": 0.02, "lambda_u": 0.0, "lambda_v": 0.0, "congestion": True},
    "D3": {"sigma_p": 0.15, "lambda_b": 0.03, "lambda_u": 0.01, "lambda_v": 0.005, "congestion": True},
    "D4": {"sigma_p": 0.20, "lambda_b": 0.05, "lambda_u": 0.02, "lambda_v": 0.01, "congestion": True},
}


class DisruptionEngine:
    def __init__(self, profile_name: str, rng: np.random.RandomState):
        self.params = PROFILES[profile_name]
        self.rng = rng
        self._pending = []
        self._mu_b = 0.1
        self._mu_v = 0.2
        self._kappa = 1.5
        self._lambda_c = 2.0

    def reset(self):
        self._pending.clear()
        self._schedule_initial_events()

    def _schedule_initial_events(self):
        if self.params["lambda_b"] > 0:
            t = self.rng.exponential(1.0 / self.params["lambda_b"])
            self._pending.append((t, "breakdown"))
        if self.params["lambda_u"] > 0:
            t = self.rng.exponential(1.0 / self.params["lambda_u"])
            self._pending.append((t, "urgent_insertion"))
        if self.params["lambda_v"] > 0:
            t = self.rng.exponential(1.0 / self.params["lambda_v"])
            self._pending.append((t, "agv_failure"))
        self._pending.sort()

    def next_event_time(self, current_time):
        future = [t for t, _ in self._pending if t > current_time]
        return min(future) if future else float("inf")

    def generate_event(self, event_time):
        idx = next(i for i, (t, _) in enumerate(self._pending) if t == event_time)
        _, etype = self._pending.pop(idx)
        event = {"type": etype, "time": event_time}
        if etype == "breakdown":
            event["machine"] = self.rng.randint(0, 20)
            event["duration"] = self.rng.exponential(1.0 / self._mu_b)
            self._pending.append((event_time + event["duration"], "repair"))
            next_bd = event_time + self.rng.exponential(1.0 / self.params["lambda_b"])
            self._pending.append((next_bd, "breakdown"))
        elif etype == "urgent_insertion":
            n_ops = self.rng.randint(2, 7)
            event["job_data"] = self._generate_urgent_job(n_ops, event_time)
            next_ins = event_time + self.rng.exponential(1.0 / self.params["lambda_u"])
            self._pending.append((next_ins, "urgent_insertion"))
        elif etype == "agv_failure":
            event["agv"] = self.rng.randint(0, 4)
            event["duration"] = self.rng.exponential(1.0 / self._mu_v)
            self._pending.append((event_time + event["duration"], "agv_recovery"))
            next_fail = event_time + self.rng.exponential(1.0 / self.params["lambda_v"])
            self._pending.append((next_fail, "agv_failure"))
        self._pending.sort()
        return event

    def _generate_urgent_job(self, n_ops, arrival_time):
        ops = []
        for _ in range(n_ops):
            n_eligible = self.rng.randint(2, 6)
            eligible = self.rng.choice(20, n_eligible, replace=False).tolist()
            ptimes = {m: float(self.rng.uniform(5, 50)) for m in eligible}
            ops.append({"eligible_machines": eligible, "processing_times": ptimes})
        mean_p = np.mean([np.mean(list(o["processing_times"].values())) for o in ops])
        due_date = arrival_time + self._kappa * mean_p * n_ops
        return {"operations": ops, "due_date": due_date, "weight": 1.0}

    def perturb_processing_time(self, base_time):
        if self.params["sigma_p"] > 0:
            eps = self.rng.normal(0, self.params["sigma_p"])
            return base_time * (1 + eps)
        return base_time

    def compute_congestion_delay(self, queue_length):
        if self.params["congestion"]:
            return self._lambda_c * queue_length
        return 0.0
