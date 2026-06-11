import math

import torch
import torch_geometric

from torch import nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GraphConv, EdgeConv, GCNConv, MessagePassing

from torch_cluster import radius_graph, knn_graph

N_EDGE_FEATURES = 3   # ln(dR), ln(kT), ln(z)


class EdgeFeatureConv(MessagePassing):
    def __init__(self, nn):
        super(EdgeFeatureConv, self).__init__(aggr='max')
        self.nn = nn

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.nn(torch.cat([x_i, x_j - x_i, edge_attr], dim=-1))


class GraphMETNetworkEdgeFeatures(nn.Module):
    def __init__ (self, continuous_dim, cat_dim, norm, output_dim=1, hidden_dim=32, conv_depth=1, use_edge_features=False):
        super(GraphMETNetworkEdgeFeatures, self).__init__()

        self.datanorm = norm
        self.use_edge_features = use_edge_features

        self.embed_charge = nn.Embedding(3, hidden_dim//4)
        self.embed_pdgid = nn.Embedding(7, hidden_dim//4)
        #self.embed_pv = nn.Embedding(8, hidden_dim//4)

        self.embed_continuous = nn.Sequential(nn.Linear(continuous_dim,hidden_dim//2),
                                              nn.ELU(),
                                              #nn.BatchNorm1d(hidden_dim//2) # uncomment if it starts overtraining
                                             )

        self.embed_categorical = nn.Sequential(nn.Linear(2*hidden_dim//4,hidden_dim//2),
                                               nn.ELU(),
                                               #nn.BatchNorm1d(hidden_dim//2)
                                              )

        self.encode_all = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                        nn.ELU()
                                       )
        self.bn_all = nn.BatchNorm1d(hidden_dim)

        self.conv_continuous = nn.ModuleList()
        for i in range(conv_depth):
            self.conv_continuous.append(nn.ModuleList())
            if self.use_edge_features:
                mesg = nn.Sequential(nn.Linear(2*hidden_dim + N_EDGE_FEATURES, hidden_dim))
                self.conv_continuous[-1].append(EdgeFeatureConv(nn=mesg))
            else:
                mesg = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim))
                self.conv_continuous[-1].append(EdgeConv(nn=mesg).jittable())
            self.conv_continuous[-1].append(nn.BatchNorm1d(hidden_dim))

        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2),
                                    nn.ELU(),
                                    nn.Linear(hidden_dim//2, output_dim)
                                   )
        self.pdgs = [1, 2, 11, 13, 22, 130, 211]

    def compute_edge_features(self, x_cont, edge_index, eps=1e-8):
        # raw physical units: pt at index 0, eta at index 3, phi at index 4
        src, dst = edge_index[0], edge_index[1]
        pt_i, pt_j = x_cont[src, 0], x_cont[dst, 0]
        eta_i, eta_j = x_cont[src, 3], x_cont[dst, 3]
        phi_i, phi_j = x_cont[src, 4], x_cont[dst, 4]

        # wrap phi difference into (-pi, pi]
        dphi = (phi_i - phi_j + math.pi) % (2 * math.pi) - math.pi
        dr = torch.sqrt((eta_i - eta_j)**2 + dphi**2)
        ptmin = torch.minimum(pt_i, pt_j)

        ln_dr = torch.log(dr.clamp(min=eps))
        ln_kt = torch.log((ptmin * dr).clamp(min=eps))
        ln_z = torch.log((ptmin / (pt_i + pt_j).clamp(min=eps)).clamp(min=eps))

        return torch.stack([ln_dr, ln_kt, ln_z], dim=1)

    def forward(self, x_cont, x_cat, edge_index, batch):
        # Normalize the input values within [0,1] range: pt, px, py, eta, phi, puppiWeight, pdgId, charge
        #norm = torch.tensor([1./2950., 1./2950, 1./2950, 1., 1., 1.]).to(device)

        # edge features use raw (physical) pt, eta, phi -- compute before normalization
        if self.use_edge_features:
            edge_attr = self.compute_edge_features(x_cont, edge_index)

        x_cont *= self.datanorm

        emb_cont = self.embed_continuous(x_cont)
        emb_chrg = self.embed_charge(x_cat[:, 1] + 1)
        #emb_pv = self.embed_pv(x_cat[:, 2])

        pdg_remap = torch.abs(x_cat[:, 0])
        for i, pdgval in enumerate(self.pdgs):
            pdg_remap = torch.where(pdg_remap == pdgval, torch.full_like(pdg_remap, i), pdg_remap)
        emb_pdg = self.embed_pdgid(pdg_remap)

        emb_cat = self.embed_categorical(torch.cat([emb_chrg, emb_pdg], dim=1))
        #emb_cat = self.embed_categorical(torch.cat([emb_chrg, emb_pdg, emb_pv], dim=1))
        emb = self.bn_all(self.encode_all(torch.cat([emb_cat, emb_cont], dim=1)))

        # graph convolution for continuous variables
        for co_conv in self.conv_continuous:
            #dynamic, evolving knn
            #emb = emb + co_conv[1](co_conv[0](emb, knn_graph(emb, k=20, batch=batch, loop=True)))
            #static
            if self.use_edge_features:
                emb = emb + co_conv[1](co_conv[0](emb, edge_index, edge_attr))
            else:
                emb = emb + co_conv[1](co_conv[0](emb, edge_index))

        out = self.output(emb)

        return out.squeeze(-1)
