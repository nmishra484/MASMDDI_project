import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, TransformerConv


# =====================================================
# ----------------- CROSS DRUG ATTENTION --------------
# =====================================================

class CrossDrugAttention(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.w_q = nn.Linear(n_features, n_features)
        self.w_k = nn.Linear(n_features, n_features)

    def forward(self, h1, h2):
        q = self.w_q(h1)
        k = self.w_k(h2)

        # Improved interaction (element-wise)
        interaction = torch.tanh(q * k)
        return interaction


# =====================================================
# ------------- GEOMETRY WARP SCORING ------------------
# =====================================================

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
# ------------------ GRAPH ENCODER ---------------------
# =====================================================

class MASMG(nn.Module):
    def __init__(self, in_features, hidden_dim, num_layers):
        super().__init__()

        self.lin0 = nn.Linear(in_features, hidden_dim)

        self.convs = nn.ModuleList([
            TransformerConv(hidden_dim, hidden_dim, heads=4, concat=False)
            for _ in range(num_layers)
        ])

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Initial projection
        x = self.lin0(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Transformer layers
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.2, training=self.training)

        # Graph-level embedding
        graph_feat = global_add_pool(x, batch)

        return graph_feat


# =====================================================
# -------------------- MAIN MODEL ----------------------
# =====================================================

class MASMDDI(nn.Module):
    def __init__(self, in_features, hidden_dim, rel_total, num_layers):
        super().__init__()

        self.encoder = MASMG(in_features, hidden_dim, num_layers)

        self.cross_attention = CrossDrugAttention(hidden_dim)
        self.scorer = RelationalGeometryWarp(rel_total, hidden_dim)

    def forward(self, triples):
        HData, TData, Rels = triples

        # Encode both drugs
        h_embed = self.encoder(HData)
        t_embed = self.encoder(TData)

        # Cross-drug interaction
        interaction_feat = self.cross_attention(h_embed, t_embed)

        # Geometry-aware scoring
        scores = self.scorer(
            h_embed + interaction_feat,
            t_embed + interaction_feat,
            Rels
        )

        return scores.squeeze()