import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import (
    global_add_pool,
    GCNConv
)

# =====================================================
# CROSS DRUG ATTENTION
# =====================================================

class CrossDrugAttention(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.w_q = nn.Linear(
            dim,
            dim
        )

        self.w_k = nn.Linear(
            dim,
            dim
        )

    def forward(
        self,
        h1,
        h2
    ):

        q = self.w_q(h1)

        k = self.w_k(h2)

        interaction = torch.tanh(
            q * k
        )

        return interaction


# =====================================================
# GEOMETRY WARP
# =====================================================

class RelationalGeometryWarp(nn.Module):

    def __init__(
        self,
        n_rels,
        dim
    ):

        super().__init__()

        self.dim = dim

        self.rel = nn.Embedding(
            n_rels,
            dim * dim
        )

        nn.init.xavier_uniform_(
            self.rel.weight
        )

    def forward(
        self,
        h,
        t,
        r
    ):

        R = self.rel(r).view(
            -1,
            self.dim,
            self.dim
        )

        R = F.normalize(
            R,
            dim=-1
        )

        h = F.normalize(
            h,
            dim=-1
        ).unsqueeze(1)

        t = F.normalize(
            t,
            dim=-1
        ).unsqueeze(-1)

        score = torch.matmul(
            torch.matmul(h, R),
            t
        )

        return score.squeeze()


# =====================================================
# GRAPH ENCODER
# =====================================================

class Encoder(nn.Module):

    def __init__(
        self,
        in_dim,
        hid_dim,
        layers=4
    ):

        super().__init__()

        self.lin = nn.Linear(
            in_dim,
            hid_dim
        )

        # =========================================
        # GCNConv Instead of TransformerConv
        # =========================================

        self.convs = nn.ModuleList([

            GCNConv(
                hid_dim,
                hid_dim
            )

            for _ in range(layers)

        ])

    def forward(self, data):

        x = data.x

        edge_index = data.edge_index

        batch = data.batch

        x = self.lin(x)

        x = F.dropout(
            x,
            0.3,
            training=self.training
        )

        for conv in self.convs:

            x = F.relu(
                conv(
                    x,
                    edge_index
                )
            )

            x = F.dropout(
                x,
                0.3,
                training=self.training
            )

        graph_embedding = global_add_pool(
            x,
            batch
        )

        return graph_embedding


# =====================================================
# CONTRASTIVE PROJECTION HEAD
# =====================================================

class ProjectionHead(nn.Module):

    def __init__(
        self,
        dim
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                dim,
                dim
            ),

            nn.ReLU(),

            nn.Linear(
                dim,
                dim
            )
        )

    def forward(self, x):

        z = self.net(x)

        z = F.normalize(
            z,
            dim=-1
        )

        return z


# =====================================================
# MAIN MODEL
# =====================================================

class MASMDDI(nn.Module):

    def __init__(
        self,
        in_dim,
        hid_dim,
        rel_total
    ):

        super().__init__()

        self.encoder = Encoder(
            in_dim,
            hid_dim
        )

        self.attn = CrossDrugAttention(
            hid_dim
        )

        self.scorer = RelationalGeometryWarp(
            rel_total,
            hid_dim
        )

        self.projector = ProjectionHead(
            hid_dim
        )

    def forward(self, triples):

        H, T, R = triples

        # =====================================
        # ENCODER
        # =====================================

        h = self.encoder(H)

        t = self.encoder(T)

        # =====================================
        # CROSS DRUG ATTENTION
        # =====================================

        interaction = self.attn(
            h,
            t
        )

        h_fused = h + interaction

        t_fused = t + interaction

        # =====================================
        # RELATIONAL GEOMETRY WARP
        # =====================================

        score = self.scorer(
            h_fused,
            t_fused,
            R
        )

        # =====================================
        # CONTRASTIVE PROJECTION
        # =====================================

        z_h = self.projector(
            h_fused
        )

        z_t = self.projector(
            t_fused
        )

        return score, z_h, z_t
