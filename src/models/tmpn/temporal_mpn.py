import torch
import torch.nn as nn
from .gru_gate import NodeGRUGate
from .typed_aggregator import TypedMeanAggregator
from .spectral_norm import apply_spectral_norm_to_relation_weights


class TemporalMessagePassingNetwork(nn.Module):
    """L-layer heterogeneous message passing with per-node GRU temporal gating.

    Implements Equations (6)-(10).  Hidden states are NOT reset between
    decision epochs so that h^{(l)}_i at epoch t depends on the full
    history G_1, ..., G_t.
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 4,
                 num_edge_types: int = 6, num_node_types: int = 3,
                 input_dims: dict | None = None, spectral_bound: float = 1.0,
                 dropout: float = 0.0):
        super().__init__()
        self.L = num_layers
        self.d = hidden_dim
        self.num_edge_types = num_edge_types
        self._spectral_bound = spectral_bound

        _default_input = {"operation": 8, "machine": 7, "agv": 7}
        input_dims = input_dims or _default_input

        self.input_projs = nn.ModuleDict({
            ntype: nn.Linear(dim, hidden_dim)
            for ntype, dim in input_dims.items()
        })

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = _TMPNLayer(hidden_dim, num_edge_types, dropout)
            self.layers.append(layer)

        self._apply_spectral_norm()

    def _apply_spectral_norm(self):
        for layer in self.layers:
            apply_spectral_norm_to_relation_weights(
                layer.aggregator, bound=self._spectral_bound
            )

    def forward(self, node_features: dict[str, torch.Tensor],
                edge_index_dict: dict[str, torch.Tensor],
                hidden_states: list[dict[str, torch.Tensor]] | None = None):
        """
        Parameters
        ----------
        node_features : {node_type: (N_type, d_type)}
        edge_index_dict : {edge_type_str: (2, E_r)}
        hidden_states : per-layer per-type hidden from previous epoch

        Returns
        -------
        embeddings : {node_type: (N_type, d)}
        new_hidden : list[dict[str, Tensor]]
        """
        h = {}
        for ntype, x in node_features.items():
            h[ntype] = self.input_projs[ntype](x)

        if hidden_states is None:
            hidden_states = [
                {nt: torch.zeros_like(h[nt]) for nt in h}
                for _ in range(self.L)
            ]

        new_hidden = []
        for l, layer in enumerate(self.layers):
            h, h_new = layer(h, edge_index_dict, hidden_states[l])
            new_hidden.append(h_new)

        return h, new_hidden

    def readout(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """Type-wise mean pooling -> concat  (Equation 11)."""
        parts = []
        for ntype in ["operation", "machine", "agv"]:
            if ntype in embeddings and embeddings[ntype].size(0) > 0:
                parts.append(embeddings[ntype].mean(dim=0))
            else:
                parts.append(torch.zeros(self.d, device=next(iter(embeddings.values())).device))
        return torch.cat(parts, dim=-1)


class _TMPNLayer(nn.Module):
    def __init__(self, d, num_edge_types, dropout):
        super().__init__()
        self.aggregator = TypedMeanAggregator(d, num_edge_types)
        self.gru = NodeGRUGate(d)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h, edge_index_dict, prev_hidden):
        messages = self.aggregator(h, edge_index_dict)
        new_h = {}
        new_hidden = {}
        for ntype in h:
            m = messages.get(ntype, torch.zeros_like(h[ntype]))
            m = self.dropout(m)
            new_h[ntype], new_hidden[ntype] = self.gru(m, h[ntype], prev_hidden[ntype])
        return new_h, new_hidden
