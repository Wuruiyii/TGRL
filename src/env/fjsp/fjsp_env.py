import numpy as np
from ..agv.agv_manager import AGVManager
from ..disruptions.disruption_engine import DisruptionEngine
from ...graph.construct.builder import HeteroGraphBuilder


class FJSPAGVEnv:
    """Simulates the FJSP-AGV environment with event-driven stepping.

    State: heterogeneous graph G_t with operation/machine/AGV nodes.
    Action: (operation, machine, AGV) triple.
    Reward: Eq. (5) composite signal.
    """

    def __init__(self, instance, disruption_profile="D0", seed=None,
                 reward_weights=None):
        self.instance = instance
        self.rng = np.random.RandomState(seed)
        self._rw = reward_weights or {
            "alpha1": 1.0, "alpha2": 0.5, "alpha3": 0.3, "alpha4": 10.0
        }
        self.gamma = 0.99
        self.disruption_engine = DisruptionEngine(disruption_profile, self.rng)
        self.agv_manager = AGVManager(instance["agvs"], instance["guidepath"])
        self.graph_builder = HeteroGraphBuilder(instance)

        self._time = 0.0
        self._done = False
        self._operations = None
        self._machines = None

    def reset(self):
        self._time = 0.0
        self._done = False
        self._init_operations()
        self._init_machines()
        self.agv_manager.reset()
        self.disruption_engine.reset()
        return self._build_state()

    def step(self, action):
        op_idx, m_idx, v_idx = action["operation"], action["machine"], action["agv"]
        self._execute_dispatch(op_idx, m_idx, v_idx)
        events = self._advance_to_next_epoch()
        for ev in events:
            self._handle_event(ev)
        reward = self._compute_reward()
        state = self._build_state()
        return state, reward, self._done, {"time": self._time}

    def _init_operations(self):
        self._operations = []
        for j, job in enumerate(self.instance["jobs"]):
            for i, op in enumerate(job["operations"]):
                self._operations.append({
                    "job": j, "op_idx": i,
                    "status": "unscheduled",
                    "eligible": op["eligible_machines"],
                    "proc_times": op["processing_times"],
                    "remaining": 0.0,
                    "assigned_machine": None,
                    "completion_time": None,
                })

    def _init_machines(self):
        self._machines = [
            {"status": "idle", "queue": [], "utilisation": 0.0,
             "breakdown_history": False, "current_op": None}
            for _ in range(self.instance["num_machines"])
        ]

    def _execute_dispatch(self, op_idx, m_idx, v_idx):
        op = self._operations[op_idx]
        op["status"] = "waiting"
        op["assigned_machine"] = m_idx
        transport_time = self.agv_manager.dispatch(v_idx, op, m_idx)
        op["remaining"] = op["proc_times"][m_idx] + transport_time

    def _advance_to_next_epoch(self):
        events = []
        min_time = float("inf")
        for op in self._operations:
            if op["status"] == "in_process" and op["remaining"] < min_time:
                min_time = op["remaining"]
        disruption_time = self.disruption_engine.next_event_time(self._time)
        if disruption_time < min_time:
            min_time = disruption_time
            events.append(self.disruption_engine.generate_event(disruption_time))
        dt = max(min_time, 0.001)
        self._time += dt
        for op in self._operations:
            if op["status"] == "in_process":
                op["remaining"] -= dt
                if op["remaining"] <= 0:
                    op["status"] = "completed"
                    op["completion_time"] = self._time
                    events.append({"type": "completion", "op": op})
        if all(op["status"] == "completed" for op in self._operations):
            self._done = True
        return events

    def _handle_event(self, event):
        etype = event["type"]
        if etype == "breakdown":
            m = event["machine"]
            self._machines[m]["status"] = "broken"
            self._machines[m]["breakdown_history"] = True
        elif etype == "repair":
            self._machines[event["machine"]]["status"] = "idle"
        elif etype == "urgent_insertion":
            self._insert_job(event["job_data"])
        elif etype == "agv_failure":
            self.agv_manager.fail_agv(event["agv"])

    def _insert_job(self, job_data):
        j = len(self.instance["jobs"])
        for i, op in enumerate(job_data["operations"]):
            self._operations.append({
                "job": j, "op_idx": i, "status": "unscheduled",
                "eligible": op["eligible_machines"],
                "proc_times": op["processing_times"],
                "remaining": 0.0, "assigned_machine": None,
                "completion_time": None,
            })

    def _compute_reward(self):
        cmax_delta = self._estimate_makespan_delta()
        tardiness = sum(
            max(0, op["completion_time"] - self.instance["jobs"][op["job"]].get("due_date", 1e9))
            for op in self._operations if op["completion_time"] is not None
        )
        idle_frac = np.mean([1.0 if m["status"] == "idle" else 0.0 for m in self._machines])
        completion_bonus = 10.0 if self._done else 0.0
        r = (-self._rw["alpha1"] * cmax_delta
             - self._rw["alpha2"] * tardiness
             - self._rw["alpha3"] * idle_frac
             + self._rw["alpha4"] * completion_bonus)
        return r

    def _estimate_makespan_delta(self):
        completed = [op for op in self._operations if op["completion_time"] is not None]
        if not completed:
            return 0.0
        return max(op["completion_time"] for op in completed) - self._time

    def _build_state(self):
        return self.graph_builder.build(
            self._operations, self._machines,
            self.agv_manager.get_states(), self._time
        )
