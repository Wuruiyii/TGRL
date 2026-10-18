import numpy as np


class HeteroGraphBuilder:
    """Constructs G_t = (V_t, E_t, X_t) with three node types and six edge types."""

    EDGE_TYPES = ["precedence", "eligibility", "assignment",
                  "transport_request", "guidepath_adj", "congestion"]

    def __init__(self, instance):
        self.instance = instance
        self._d_O = 8
        self._d_M = 7
        self._d_V = 7

    def build(self, operations, machines, agv_states, current_time):
        node_features = {
            "operation": self._build_op_features(operations, current_time),
            "machine": self._build_machine_features(machines, current_time),
            "agv": self._build_agv_features(agv_states),
        }
        edge_index_dict = self._build_edges(operations, machines, agv_states)
        masks = self._build_masks(operations, machines, agv_states)
        return {
            "node_features": node_features,
            "edge_index_dict": edge_index_dict,
            "masks": masks,
            "time": current_time,
        }

    def _build_op_features(self, operations, t):
        feats = []
        for op in operations:
            status_oh = [0.0] * 4
            status_map = {"unscheduled": 0, "waiting": 1, "in_process": 2, "completed": 3}
            status_oh[status_map.get(op["status"], 0)] = 1.0
            ptimes = list(op["proc_times"].values())
            mean_p = np.mean(ptimes) if ptimes else 0.0
            var_p = np.var(ptimes) if ptimes else 0.0
            remaining = op["remaining"]
            n_ops = len(self.instance["jobs"][op["job"]]["operations"])
            pos_norm = op["op_idx"] / max(n_ops - 1, 1)
            dd = self.instance["jobs"][op["job"]].get("due_date", 1e6)
            slack = (dd - t) / max(dd, 1e-6)
            feats.append(status_oh + [mean_p / 50.0, var_p / 625.0, remaining / 50.0, pos_norm, slack])
        import torch
        return torch.tensor(feats, dtype=torch.float32) if feats else torch.zeros(0, self._d_O)

    def _build_machine_features(self, machines, t):
        feats = []
        for m in machines:
            status_oh = [0.0] * 3
            status_map = {"idle": 0, "busy": 1, "broken": 2}
            status_oh[status_map.get(m["status"], 0)] = 1.0
            util = m.get("utilisation", 0.0)
            qlen = len(m.get("queue", [])) / 10.0
            eta = 0.0
            bd_hist = 1.0 if m.get("breakdown_history", False) else 0.0
            feats.append(status_oh + [util, qlen, eta, bd_hist])
        import torch
        return torch.tensor(feats, dtype=torch.float32)

    def _build_agv_features(self, agv_states):
        feats = []
        for a in agv_states:
            pos = a.get("position", (0, 0))
            px, py = pos[0] / 5.0, pos[1] / 4.0
            status_oh = [0.0] * 4
            status_map = {"idle": 0, "loaded": 1, "empty_travel": 2, "failed": 3}
            status_oh[status_map.get(a["status"], 0)] = 1.0
            battery = a.get("battery", 1.0)
            cong = 1.0 if a.get("congestion_flag", False) else 0.0
            feats.append([px, py] + status_oh + [battery, cong])
        import torch
        return torch.tensor(feats, dtype=torch.float32) if feats else torch.zeros(0, self._d_V)

    def _build_edges(self, operations, machines, agv_states):
        edges = {}
        prec_src, prec_dst = [], []
        for op in operations:
            j, i = op["job"], op["op_idx"]
            if i > 0:
                prev_idx = self._find_op_index(operations, j, i - 1)
                cur_idx = self._find_op_index(operations, j, i)
                if prev_idx is not None and cur_idx is not None:
                    prec_src.append(prev_idx)
                    prec_dst.append(cur_idx)
        import torch
        edges["operation__precedence__operation"] = torch.tensor([prec_src, prec_dst], dtype=torch.long) if prec_src else torch.zeros(2, 0, dtype=torch.long)

        elig_src, elig_dst = [], []
        for idx, op in enumerate(operations):
            for m in op["eligible"]:
                elig_src.append(idx)
                elig_dst.append(m)
        edges["operation__eligibility__machine"] = torch.tensor([elig_src, elig_dst], dtype=torch.long) if elig_src else torch.zeros(2, 0, dtype=torch.long)

        assign_src, assign_dst = [], []
        for idx, op in enumerate(operations):
            if op["assigned_machine"] is not None:
                assign_src.append(idx)
                assign_dst.append(op["assigned_machine"])
        edges["operation__assignment__machine"] = torch.tensor([assign_src, assign_dst], dtype=torch.long) if assign_src else torch.zeros(2, 0, dtype=torch.long)

        edges["machine__transport_request__agv"] = torch.zeros(2, 0, dtype=torch.long)
        edges["agv__guidepath_adj__agv"] = torch.zeros(2, 0, dtype=torch.long)
        edges["agv__congestion__machine"] = torch.zeros(2, 0, dtype=torch.long)
        return edges

    @staticmethod
    def _find_op_index(operations, job, op_idx):
        for i, op in enumerate(operations):
            if op["job"] == job and op["op_idx"] == op_idx:
                return i
        return None

    def _build_masks(self, operations, machines, agv_states):
        import torch
        sched = torch.tensor([op["status"] == "unscheduled" for op in operations])
        elig = torch.tensor([m["status"] != "broken" for m in machines])
        avail = torch.tensor([a["status"] == "idle" for a in agv_states])
        return {"schedulable_ops": sched, "eligible_machines": elig, "available_agvs": avail}
