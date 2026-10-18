import torch
import torch.nn as nn
import torch.nn.functional as F
from ..tmpn import TemporalMessagePassingNetwork
from ..heads.operation_head import OperationHead
from ..heads.machine_head import MachineHead
from ..heads.agv_head import AGVHead
from .gae import compute_gae


class PPOAgent(nn.Module):
    """Hierarchical PPO agent with separate value head.

    Action decomposition (Eq. 12):
        pi(a|s) = pi_O(o|s) * pi_M(m|s,o) * pi_V(v|s,o,m)
    """

    def __init__(self, config):
        super().__init__()
        d = config.get("hidden_dim", 128)
        self.tmpn = TemporalMessagePassingNetwork(
            hidden_dim=d,
            num_layers=config.get("num_layers", 4),
            num_edge_types=config.get("num_edge_types", 6),
            spectral_bound=config.get("spectral_bound", 1.0),
        )
        self.op_head = OperationHead(d)
        self.machine_head = MachineHead(d)
        self.agv_head = AGVHead(d)
        self.value_head = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.clip_eps = config.get("clip_eps", 0.2)
        self.entropy_coeff = config.get("entropy_coeff", 0.01)
        self.value_coeff = config.get("value_coeff", 0.5)

    def forward(self, state, hidden=None, deterministic=False):
        node_feat = state["node_features"]
        edge_idx = state["edge_index_dict"]
        masks = state["masks"]

        embeddings, new_hidden = self.tmpn(node_feat, edge_idx, hidden)
        g = self.tmpn.readout(embeddings)

        op_logits, op_idx = self.op_head(g, embeddings["operation"],
                                          masks["schedulable_ops"],
                                          deterministic)
        op_emb = embeddings["operation"][op_idx]

        m_logits, m_idx = self.machine_head(g, embeddings["machine"],
                                             op_emb, masks["eligible_machines"],
                                             deterministic)

        v_logits, v_idx = self.agv_head(g, embeddings["agv"], op_emb,
                                         embeddings["machine"][m_idx],
                                         masks["available_agvs"],
                                         deterministic)

        value = self.value_head(g).squeeze(-1)
        log_prob = (F.log_softmax(op_logits, -1).gather(-1, op_idx.unsqueeze(-1)).squeeze(-1)
                    + F.log_softmax(m_logits, -1).gather(-1, m_idx.unsqueeze(-1)).squeeze(-1)
                    + F.log_softmax(v_logits, -1).gather(-1, v_idx.unsqueeze(-1)).squeeze(-1))

        action = {"operation": op_idx, "machine": m_idx, "agv": v_idx}
        return action, log_prob, value, new_hidden

    def evaluate(self, states, actions, hiddens):
        """Evaluate log_prob and value for a batch of stored transitions."""
        log_probs, values, entropies = [], [], []
        h = hiddens[0] if hiddens else None
        for i, (s, a) in enumerate(zip(states, actions)):
            _, lp, v, h = self.forward(s, h, deterministic=False)
            log_probs.append(lp)
            values.append(v)
        return torch.stack(log_probs), torch.stack(values), None

    def ppo_loss(self, old_log_probs, new_log_probs, advantages, values,
                 returns):
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns)
        return policy_loss + self.value_coeff * value_loss
