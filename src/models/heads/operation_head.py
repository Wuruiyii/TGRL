import torch
import torch.nn as nn
import torch.nn.functional as F


class OperationHead(nn.Module):
    """Eq. 13: attention over schedulable operations conditioned on g_t."""

    def __init__(self, d: int):
        super().__init__()
        self.W_O = nn.Linear(4 * d, d, bias=False)
        self.q_O = nn.Parameter(torch.randn(d))
        nn.init.xavier_uniform_(self.W_O.weight)

    def forward(self, g, op_embeddings, mask, deterministic=False):
        N = op_embeddings.size(0)
        g_exp = g.unsqueeze(0).expand(N, -1)
        combined = torch.cat([g_exp, op_embeddings], dim=-1)
        scores = torch.matmul(torch.tanh(self.W_O(combined)), self.q_O)
        scores = scores.masked_fill(~mask, float("-inf"))
        logits = scores
        if deterministic:
            idx = logits.argmax()
        else:
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).squeeze(-1)
        return logits, idx
