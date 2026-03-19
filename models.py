import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, TransformerConv


# =====================================================
# ----------------- NOVELTY MODULES -------------------
# =====================================================

class RelationalHyperNet(nn.Module):
    def __init__(self, n_features, n_rels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(n_rels, n_features),
            nn.Linear(n_features, n_features),
            nn.LeakyReLU(0.2),
            nn.Linear(n_features, n_features),
            nn.Sigmoid()
        )

    def forward(self, x, rel_idx):
        gate = self.net(rel_idx)
        return x * gate


class SpectralGlobalFilter(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.filter = nn.Parameter(torch.ones(n_features))
        self.lin = nn.Linear(n_features, n_features)

    def forward(self, x, batch):
        g_context = global_add_pool(x, batch)
        spectral_x = g_context * self.filter
        return torch.tanh(self.lin(spectral_x))


class CrossModalityGate(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_features * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, spatial_feat, spectral_feat):
        combined = torch.cat([spatial_feat, spectral_feat], dim=-1)
        g = self.gate(combined)
        return g * spatial_feat + (1 - g) * spectral_feat


class MultiHeadMolecularAttention(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.w_q = nn.Linear(n_features, n_features)
        self.w_k = nn.Linear(n_features, n_features)

    def forward(self, h1, h2):
        q = self.w_q(h1)
        k = self.w_k(h2)
        return torch.tanh(q + k)


class RelationalGeometryWarp(nn.Module):
    def __init__(self, n_rels, n_features):
        super().__init__()
        self.n_features = n_features
        self.rel_emb = nn.Embedding(n_rels, n_features * n_features)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, heads, tails, rels):
        rel_mat = self.rel_emb(rels).view(-1, self.n_features, self.n_features)
        rel_mat = F.normalize(rel_mat, dim=-1)

        h = F.normalize(heads, dim=-1).unsqueeze(1)
        t = F.normalize(tails, dim=-1).unsqueeze(-1)

        score = torch.matmul(torch.matmul(h, rel_mat), t)
        return score.squeeze()


# =====================================================
# ------------------ GRAPH ENCODER --------------------
# =====================================================

class MASMG(nn.Module):
    def __init__(self, in_features, hidden_dim, num_layers):
        super().__init__()

        self.lin0 = nn.Linear(in_features, hidden_dim)

        self.convs = nn.ModuleList([
            TransformerConv(hidden_dim, hidden_dim, heads=4, concat=False)
            for _ in range(num_layers)
        ])

        self.spectral = SpectralGlobalFilter(hidden_dim)
        self.fusion = CrossModalityGate(hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.lin0(x)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        spatial_feat = global_add_pool(x, batch)
        spectral_feat = self.spectral(x, batch)

        graph_feat = self.fusion(spatial_feat, spectral_feat)

        return graph_feat


# =====================================================
# -------------------- MAIN MODEL ---------------------
# =====================================================

class MASMDDI(nn.Module):
    def __init__(self, in_features, hidden_dim, rel_total, num_layers):
        super().__init__()

        self.encoder = MASMG(in_features, hidden_dim, num_layers)

        self.cross_attention = MultiHeadMolecularAttention(hidden_dim)
        self.hypernet = RelationalHyperNet(hidden_dim, rel_total)
        self.scorer = RelationalGeometryWarp(rel_total, hidden_dim)

    def forward(self, triples):
        HData, TData, Rels = triples

        h_embed = self.encoder(HData)
        t_embed = self.encoder(TData)

        interaction_feat = self.cross_attention(h_embed, t_embed)

        h_embed = self.hypernet(h_embed, Rels)
        t_embed = self.hypernet(t_embed, Rels)

        scores = self.scorer(h_embed + interaction_feat,
                             t_embed + interaction_feat,
                             Rels)

        return scores.squeeze()