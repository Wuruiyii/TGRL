import torch
import torch.nn as nn


class NodeGRUGate(nn.Module):
    """Per-node GRU gating (Equations 7-10).

    The gate matrices W_z, W_r, W_h in R^{d x 2d} are shared across
    all node types at each layer.
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.W_z = nn.Linear(2 * d, d, bias=True)
        self.W_r = nn.Linear(2 * d, d, bias=True)
        self.W_h = nn.Linear(2 * d, d, bias=True)

    def forward(self, m: torch.Tensor, h_prev: torch.Tensor,
                hidden: torch.Tensor):
        """
        m      : aggregated message  (N, d)
        h_prev : embedding from previous GNN layer output (N, d)
        hidden : carried-over hidden state from previous epoch (N, d)
        """
        cat_mh = torch.cat([m, hidden], dim=-1)
        z = torch.sigmoid(self.W_z(cat_mh))
        r = torch.sigmoid(self.W_r(cat_mh))
        cat_mr = torch.cat([m, r * hidden], dim=-1)
        h_tilde = torch.tanh(self.W_h(cat_mr))
        h_new = (1 - z) * hidden + z * h_tilde
        return h_new, h_new
