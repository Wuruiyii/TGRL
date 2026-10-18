import torch
import torch.nn as nn
import torch.nn.functional as F


class MachineHead(nn.Module):
    """Attention over eligible machines conditioned on selected operation."""

    def __init__(self, d: int):
        super().__init__()
        self.W_M = nn.Linear(4 * d + d, d, bias=False)
        self.q_M = nn.Parameter(torch.randn(d))

    def forward(self, g, machine_embeddings, op_emb, mask, deterministic=False):
        N = machine_embeddings.size(0)
        g_exp = g.unsqueeze(0).expand(N, -1)
        o_exp = op_emb.unsqueeze(0).expand(N, -1)
        combined = torch.cat([g_exp, machine_embeddings, o_exp], dim=-1)
        scores = torch.matmul(torch.tanh(self.W_M(combined)), self.q_M)
        scores = scores.masked_fill(~mask, float("-inf"))
        if deterministic:
            idx = scores.argmax()
        else:
            probs = F.softmax(scores, dim=-1)
            idx = torch.multinomial(probs, 1).squeeze(-1)
        return scores, idx
