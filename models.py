import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, TransformerConv


# ---------------- CROSS-DRUG ATTENTION ----------------
class CrossDrugAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)

    def forward(self, h1, h2):
        q = self.w_q(h1)
        k = self.w_k(h2)
        return torch.tanh(q * k)


# ---------------- GEOMETRY WARP ----------------
class RelationalGeometryWarp(nn.Module):
    def __init__(self, n_rels, dim):
        super().__init__()
        self.dim = dim
        self.rel = nn.Embedding(n_rels, dim * dim)
        nn.init.xavier_uniform_(self.rel.weight)

    def forward(self, h, t, r):
        R = self.rel(r).view(-1, self.dim, self.dim)
        R = F.normalize(R, dim=-1)

        h = F.normalize(h, dim=-1).unsqueeze(1)
        t = F.normalize(t, dim=-1).unsqueeze(-1)

        score = torch.matmul(torch.matmul(h, R), t)
        return score.squeeze()


# ---------------- GRAPH ENCODER ----------------
class Encoder(nn.Module):
    def __init__(self, in_dim, hid_dim, layers=3):
        super().__init__()

        self.lin = nn.Linear(in_dim, hid_dim)

        self.convs = nn.ModuleList([
            TransformerConv(hid_dim, hid_dim, heads=4, concat=False)
            for _ in range(layers)
        ])

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.lin(x)
        x = F.dropout(x, 0.3, training=self.training)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, 0.3, training=self.training)

        return global_add_pool(x, batch)


# ---------------- MAIN MODEL ----------------
class MASMDDI(nn.Module):
    def __init__(self, in_dim, hid_dim, rel_total):
        super().__init__()

        self.encoder = Encoder(in_dim, hid_dim)
        self.attn = CrossDrugAttention(hid_dim)
        self.scorer = RelationalGeometryWarp(rel_total, hid_dim)

    def forward(self, triples):
        H, T, R = triples

        h = self.encoder(H)
        t = self.encoder(T)

        interaction = self.attn(h, t)

        return self.scorer(h + interaction, t + interaction, R)