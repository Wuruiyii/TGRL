import heapq


def shortest_path_time(guidepath, src, dst):
    """Dijkstra on guide-path graph."""
    if src == dst:
        return 0.0
    nodes = guidepath["nodes"]
    edges = guidepath["edges"]
    adj = {n: [] for n in nodes}
    for e in edges:
        adj[e["from"]].append((e["to"], e["travel_time"]))

    dist = {n: float("inf") for n in nodes}
    src_key = _pos_to_key(src, nodes)
    dst_key = _pos_to_key(dst, nodes)
    dist[src_key] = 0.0
    pq = [(0.0, src_key)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst_key:
            return d
        if d > dist[u]:
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return float("inf")


def _pos_to_key(pos, nodes):
    best = None
    best_d = float("inf")
    for n in nodes:
        d = abs(n[0] - pos[0]) + abs(n[1] - pos[1])
        if d < best_d:
            best_d = d
            best = tuple(n) if isinstance(n, list) else n
    return best
