import torch
import torch.nn as nn


class TypedMeanAggregator(nn.Module):
    """Relation-specific mean aggregation (Equation 6).

    For each edge type r, a weight matrix W_r in R^{d x d} transforms
    neighbour embeddings before mean-pooling.
    """

    _EDGE_TYPE_MAP = {
        "precedence": 0,
        "eligibility": 1,
        "assignment": 2,
        "transport_request": 3,
        "guidepath_adj": 4,
        "congestion": 5,
    }

    def __init__(self, d: int, num_edge_types: int = 6):
        super().__init__()
        self.d = d
        self.W_r = nn.ParameterList([
            nn.Parameter(torch.empty(d, d)) for _ in range(num_edge_types)
        ])
        self.b_r = nn.ParameterList([
            nn.Parameter(torch.zeros(d)) for _ in range(num_edge_types)
        ])
        self._init_weights()

    def _init_weights(self):
        for w in self.W_r:
            nn.init.xavier_uniform_(w)

    def forward(self, h: dict[str, torch.Tensor],
                edge_index_dict: dict[str, torch.Tensor]):
        all_nodes = {}
        for ntype in h:
            all_nodes[ntype] = torch.zeros_like(h[ntype])

        for etype_str, ei in edge_index_dict.items():
            if ei.size(1) == 0:
                continue
            r_idx = self._resolve_edge_type(etype_str)
            src_type, dst_type = self._parse_etype(etype_str)
            src_idx, dst_idx = ei[0], ei[1]

            src_emb = h[src_type][src_idx]
            transformed = src_emb @ self.W_r[r_idx] + self.b_r[r_idx]

            scatter_sum = torch.zeros_like(h[dst_type])
            scatter_sum.index_add_(0, dst_idx, transformed)
            counts = torch.zeros(h[dst_type].size(0), 1, device=ei.device)
            counts.index_add_(0, dst_idx, torch.ones(dst_idx.size(0), 1, device=ei.device))
            counts = counts.clamp(min=1)
            all_nodes[dst_type] = all_nodes[dst_type] + scatter_sum / counts

        return all_nodes

    def _resolve_edge_type(self, etype_str: str) -> int:
        base = etype_str.split("__")[-1] if "__" in etype_str else etype_str
        return self._EDGE_TYPE_MAP.get(base, 0)

    @staticmethod
    def _parse_etype(etype_str: str):
        parts = etype_str.split("__")
        if len(parts) == 3:
            return parts[0], parts[2]
        return "operation", "operation"


class TypedEdgeConv(nn.Module):
    """Edge-conditioned convolution variant with edge feature transform."""

    def __init__(self, d: int, edge_feat_dim: int = 1):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        self.node_mlp = nn.Linear(2 * d, d)

    def forward(self, src_emb, dst_emb, edge_feat):
        e = self.edge_mlp(edge_feat)
        out = self.node_mlp(torch.cat([src_emb * e, dst_emb], dim=-1))
        return out
