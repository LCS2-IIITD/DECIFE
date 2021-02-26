import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from dgl.nn.pytorch import GraphConv
import pickle

class SubAttn(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SubAttn, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)
        return (beta * z).sum(1)                       # (N, D * K)

class SubConv(nn.Module):
    """
    Subgraph Convolution Network.
    Arguments
    ---------
    relations : list of relations, each as a list of edge types
    edge_weights: list of edge_weights for each relation
    in_size : input feature dimension
    out_size : output feature dimension
    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, relations, edge_weights, in_size, out_size):
        super(SubConv, self).__init__()

        self.scn_layers = nn.ModuleList()
        for i in range(len(relations)):
            self.scn_layers.append(GraphConv(in_size, out_size,  norm='right', weight=True, bias=True, activation=F.elu,allow_zero_in_degree=True))
        self.scn_attention = SubAttn(in_size=out_size)
        self.relations = list(tuple(relation) for relation in relations)
        self.e =edge_weights

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        scn_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for relation in self.relations:
                self._cached_coalesced_graph[relation] = dgl.metapath_reachable_graph(
                        g, relation)
            
        for i, relation in enumerate(self.relations):
            new_g = self._cached_coalesced_graph[relation]  
            ew=torch.FloatTensor(self.e[relation])
            x=self.scn_layers[i](new_g, h, edge_weight=ew)       # (N, M, D * K)
            scn_embeddings.append(x.flatten(1))
        scn_embeddings = torch.stack(scn_embeddings, dim=1) 
        projected_embs= self.scn_attention(scn_embeddings)                            # (N, D * K)
        return projected_embs


class HSA(nn.Module):
    def __init__(self, relations,edge_weights, in_size, hidden_size, out_size, num_heads):
        super(HSA, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(SubConv(relations,edge_weights, in_size, hidden_size))
        for l in range(1,num_heads):
            self.layers.append(SubConv(relations,edge_weights, hidden_size,
                                        hidden_size))
        self.predict = nn.Linear(hidden_size, out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)
        return self.predict(h)