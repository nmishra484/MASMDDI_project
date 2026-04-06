import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch.nn import Linear, Sequential, ReLU, Sigmoid
from torch_geometric.nn import (global_add_pool, LayerNorm, JumpingKnowledge)
from conv.sparse_conv import SparseConv
from conv.weight_conv import WeightConv1, WeightConv2

from layers import (
                    CoAttentionLayer,
                    RESCAL
                    )


class MASMDDI(nn.Module):
    def __init__(self, in_features, hidd_dim, kge_dim, rel_total, num_layers, weight_conv, multi_channel):
        super().__init__()
        self.in_features = in_features
        self.hidd_dim = hidd_dim
        self.num_layers = num_layers
        self.weight_conv = weight_conv
        self.multi_channel = multi_channel
        self.rel_total = rel_total
        self.kge_dim = kge_dim
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        
        self.initial_norm = LayerNorm(self.in_features)
        self.MASMG = MASMG(self.in_features, self.num_layers, self.hidd_dim, self.weight_conv, self.multi_channel)

        self.Drug_x_max_pool = nn.MaxPool1d(self.num_layers)
        self.Drug_y_max_pool = nn.MaxPool1d(self.num_layers)
        self.attention_layer = nn.Linear(self.num_layers, self.num_layers)
        self.drug_x_attention_layer = nn.Linear(self.num_layers, self.num_layers)
        self.drug_y_attention_layer = nn.Linear(self.num_layers, self.num_layers)

        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)

    def forward(self, triples):
        HData, TData, Rels = triples

        HData.x = self.initial_norm(HData.x, HData.batch)
        TData.x = self.initial_norm(TData.x, TData.batch)

        repr_h = self.MASMG(HData)
        repr_t = self.MASMG(TData)
        kge_h = torch.stack(repr_h, dim=-2)
        kge_t = torch.stack(repr_t, dim=-2)

        attentions = self.co_attention(kge_h, kge_t)

        drug_att_x = self.drug_x_attention_layer(kge_h.permute(0, 2, 1))
        drug_att_y = self.drug_y_attention_layer(kge_t.permute(0, 2, 1))
        dx_att_layers = torch.unsqueeze(drug_att_x, 2).repeat(1, 1, kge_h.shape[-1], 1)
        dy_att_layers = torch.unsqueeze(drug_att_y, 1).repeat(1, kge_t.shape[-1], 1, 1)
        Atten_matrix = self.attention_layer(self.relu(dx_att_layers + dy_att_layers))
        Compound_x_atte = torch.mean(Atten_matrix, 2)
        Compound_y_atte = torch.mean(Atten_matrix, 1)
        Compound_x_atte = self.sigmoid(Compound_x_atte.permute(0, 2, 1))
        Compound_y_atte = self.sigmoid(Compound_y_atte.permute(0, 2, 1))
        kge_h = kge_h * 0.5 + kge_h * Compound_x_atte
        kge_t = kge_t * 0.5 + kge_t * Compound_y_atte

        # attentions = None
        scores = self.KGE(kge_h, kge_t, Rels, attentions)

        return scores


class MASMG(torch.nn.Module):
    def __init__(self,
                 in_features,
                 num_layers,
                 hidden,
                 weight_conv='WeightConv1',
                 multi_channel='False'):
        super(MASMG, self).__init__()
        self.lin0 = Linear(in_features, hidden)
        self.convs = torch.nn.ModuleList()
        self.lin1 = Linear(hidden, hidden*2)
        self.lin2 = Linear(hidden*2, hidden)
        self.jump = JumpingKnowledge(mode='max')
        for i in range(num_layers):
            self.convs.append(SparseConv(hidden, hidden))

        self.masks = torch.nn.ModuleList()
        if multi_channel == 'True':
            out_channel = hidden
        else:
            out_channel = 1
        if weight_conv != 'WeightConv2':
            for i in range(num_layers):
                self.masks.append(WeightConv1(hidden, hidden, out_channel))
        else:
            for i in range(num_layers):
                self.masks.append(WeightConv2(Sequential(
                    Linear(hidden * 2, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, out_channel),
                    Sigmoid()
                )))

    def reset_parameters(self):
        self.lin0.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for mask in self.masks:
            mask.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.lin0(x)
        mask_val = None
        xs = []
        for i, conv in enumerate(self.convs):
            mask = self.masks[i]
            mask_val = mask(x, edge_index, mask_val)
            x = F.relu(conv(x, edge_index, mask_val))
            z = F.relu(self.lin1(global_add_pool(x, batch)))
            z = F.dropout(z, p=0.5, training=self.training)
            z = F.relu(self.lin2(z))
            xs += [z]
        return xs

    def __repr__(self):
        return self.__class__.__name__
