import torch
import torch.nn as nn
import torch.nn.functional as F


class AGVHead(nn.Module):
    """Attention over available AGVs conditioned on operation and machine."""

    def __init__(self, d: int):
        super().__init__()
        self.W_V = nn.Linear(4 * d + 2 * d, d, bias=False)
        self.q_V = nn.Parameter(torch.randn(d))

    def forward(self, g, agv_embeddings, op_emb, machine_emb, mask,
                deterministic=False):
        N = agv_embeddings.size(0)
        g_exp = g.unsqueeze(0).expand(N, -1)
        o_exp = op_emb.unsqueeze(0).expand(N, -1)
        m_exp = machine_emb.unsqueeze(0).expand(N, -1)
        combined = torch.cat([g_exp, agv_embeddings, o_exp, m_exp], dim=-1)
        scores = torch.matmul(torch.tanh(self.W_V(combined)), self.q_V)
        scores = scores.masked_fill(~mask, float("-inf"))
        if deterministic:
            idx = scores.argmax()
        else:
            probs = F.softmax(scores, dim=-1)
            idx = torch.multinomial(probs, 1).squeeze(-1)
        return scores, idx
