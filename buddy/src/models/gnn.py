"""
Minimal GNN models needed by BUDDY/ELPH: SIGN and SIGNEmbedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGN(nn.Module):
    """SIGN feature transform: independent linear layers on A^k * X."""

    def __init__(self, in_channels, hidden_channels, out_channels, sign_k,
                 dropout=0.5):
        super().__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(sign_k + 1):
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.lin_out = nn.Linear((sign_k + 1) * hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, xs):
        """xs: list of tensors [X, AX, A^2X, ...]"""
        hs = []
        for x, lin, bn in zip(xs, self.lins, self.bns):
            h = F.relu(bn(lin(x)))
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)
        return self.lin_out(torch.cat(hs, dim=-1))


class SIGNEmbedding(nn.Module):
    """SIGN applied to learned node embeddings (for datasets without features)."""

    def __init__(self, in_channels, hidden_channels, out_channels, sign_k,
                 dropout=0.5):
        super().__init__()
        self.sign = SIGN(in_channels, hidden_channels, out_channels, sign_k,
                         dropout)

    def forward(self, xs):
        return self.sign(xs)
