import numpy as np
from .pathfinding import shortest_path_time


class AGVManager:
    def __init__(self, agv_configs, guidepath):
        self.num_agvs = len(agv_configs)
        self.guidepath = guidepath
        self._states = []
        for cfg in agv_configs:
            self._states.append({
                "position": cfg.get("start_pos", (0, 0)),
                "status": "idle",
                "battery": 1.0,
                "current_task": None,
                "congestion_flag": False,
            })

    def reset(self):
        for s in self._states:
            s["status"] = "idle"
            s["battery"] = 1.0
            s["current_task"] = None
            s["congestion_flag"] = False

    def dispatch(self, agv_idx, operation, machine_idx):
        s = self._states[agv_idx]
        dest = self.guidepath["machine_positions"][machine_idx]
        travel_time = shortest_path_time(self.guidepath, s["position"], dest)
        s["status"] = "loaded"
        s["current_task"] = {"op": operation, "dest": dest, "eta": travel_time}
        return travel_time

    def fail_agv(self, agv_idx):
        self._states[agv_idx]["status"] = "failed"

    def get_states(self):
        return [dict(s) for s in self._states]

    def check_congestion(self, proximity_threshold=3):
        positions = [s["position"] for s in self._states if s["status"] != "failed"]
        for i, s in enumerate(self._states):
            nearby = sum(
                1 for p in positions
                if abs(p[0] - s["position"][0]) + abs(p[1] - s["position"][1]) <= proximity_threshold
            )
            s["congestion_flag"] = nearby > 1
