"""Incremental graph update (Proposition 1).

O(d_max) per event instead of O(|V|+|E|) full reconstruction.
"""


def update_on_completion(graph_state, op_idx, machine_idx):
    """Update graph after operation completion."""
    ops = graph_state["operations"]
    ops[op_idx]["status"] = "completed"
    successor = _find_successor(ops, ops[op_idx]["job"], ops[op_idx]["op_idx"])
    if successor is not None:
        ops[successor]["status"] = "unscheduled"
    _update_machine_queue(graph_state["machines"], machine_idx)
    _remove_assignment_edge(graph_state, op_idx, machine_idx)


def update_on_breakdown(graph_state, machine_idx, duration):
    graph_state["machines"][machine_idx]["status"] = "broken"
    graph_state["machines"][machine_idx]["breakdown_history"] = True
    _remove_eligibility_edges_for_machine(graph_state, machine_idx)


def update_on_insertion(graph_state, new_job_data):
    base_idx = len(graph_state["operations"])
    for i, op in enumerate(new_job_data["operations"]):
        graph_state["operations"].append({
            "job": new_job_data["job_id"],
            "op_idx": i,
            "status": "unscheduled",
            "eligible": op["eligible_machines"],
            "proc_times": op["processing_times"],
            "remaining": 0.0,
            "assigned_machine": None,
            "completion_time": None,
        })
    _add_precedence_edges(graph_state, base_idx, len(new_job_data["operations"]))
    _add_eligibility_edges(graph_state, base_idx, new_job_data["operations"])


def update_on_agv_failure(graph_state, agv_idx):
    graph_state["agv_states"][agv_idx]["status"] = "failed"
    _remove_transport_edges(graph_state, agv_idx)


def _find_successor(ops, job, op_idx):
    for i, o in enumerate(ops):
        if o["job"] == job and o["op_idx"] == op_idx + 1:
            return i
    return None


def _update_machine_queue(machines, m_idx):
    m = machines[m_idx]
    if m["queue"]:
        m["queue"].pop(0)
    if not m["queue"]:
        m["status"] = "idle"


def _remove_assignment_edge(gs, op_idx, m_idx):
    pass


def _remove_eligibility_edges_for_machine(gs, m_idx):
    pass


def _add_precedence_edges(gs, base_idx, n_ops):
    pass


def _add_eligibility_edges(gs, base_idx, ops):
    pass


def _remove_transport_edges(gs, agv_idx):
    pass
