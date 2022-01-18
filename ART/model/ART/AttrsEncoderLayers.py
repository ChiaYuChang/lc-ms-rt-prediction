# %%
from typing import Optional, Tuple
from copy import deepcopy
import numpy as np
import torch

from ART.funcs import n_ordered_hop, hop
from torch import nn
from torch_geometric import nn as pyg_nn
from torch_geometric.data import Data
from torch.nn.init import xavier_normal_ as glorot_

class AttrsEncoderLayers(nn.Module):

    def __init__(
            self,
            num_node_attr: int,
            num_edge_attr: int,
            out_channels: int,
            hidden_channels: Optional[int] = None,
            n_hop: Optional[int] = 1,
            dropout: Optional[Tuple[float, bool]] = (0.0, False),
            direction: Optional[str] = "out"):
        
        super().__init__()

        if direction == "out":
            self._r_edge_idx = 0
        elif direction == "in":
            self._r_edge_idx = 1
        else:
            raise ValueError("direction shold be either 'out' or 'in'.")
        
        if hidden_channels is None:
            hidden_channels = out_channels

        self._n_hop = n_hop
        in_channels = num_node_attr + num_edge_attr
        self._num_node_attr = num_node_attr
        self._num_edge_attr = num_edge_attr
        self.attrs_linear_encoder = self.linear_encoder(in_channels, hidden_channels, dropout)
        self.attrs_graph_encoder = self.graph_encoder(hidden_channels, out_channels, dropout)
        self.attrs_pooling_layer = self.pooling_layer()
        self.final_btch_norm = nn.BatchNorm1d(out_channels)

    def linear_encoder(self, in_channels, out_channels, dropout):
        return  nn.Sequential(
                    nn.BatchNorm1d(
                        num_features=in_channels
                    ),
                    nn.Linear(
                        in_features=in_channels,
                        out_features=out_channels,
                        bias=False
                    ),
                    nn.BatchNorm1d(
                        num_features=out_channels
                    ),
                    nn.ReLU(),
                    nn.Dropout(p=dropout[0], inplace=dropout[1])
                )
    
    def graph_encoder(self, in_channels, out_channels, dropout):
        return  pyg_nn.GATConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout=dropout[0]
                )

    def pooling_layer(self):
        return pyg_nn.global_add_pool

    def padding_edge_attr(self, n: int, device, mu: float = 0.5, theta: float = 0.2):
        return torch.zeros((n, self._num_edge_attr), dtype=torch.float32).to(device)
        # return torch.normal(mu, theta, (n, self._num_edge_attr), dtype=torch.float32).to(device)

    def two_step_index(self, x: torch.Tensor, num_nodes: Optional[int] = None, rm_self_loop: bool = True):
        device = x.device
        y = torch.arange(start=0, end=len(x), dtype=torch.long).to(device)
        x = deepcopy(x) + len(x)
        if num_nodes is None:
            num_nodes = x.max() + 1
        
        edge_index = torch.stack((
                torch.cat((y, x), axis=0),
                torch.cat((x, y), axis=0))
        )
        edge_index = hop(edge_index, num_nodes, rm_self_loop)
        return edge_index[:, torch.all(edge_index < len(x), dim=0)]

    def n_ordered_hop(
            self, 
            edge_index: torch.Tensor,
            num_nodes: Optional[int] = None,
            rm_self_loop: Optional[bool] = True,
            n_hop: Optional[int]=2
            ) -> Tuple[torch.Tensor, int]:
        return n_ordered_hop(edge_index, num_nodes, rm_self_loop, n_hop=n_hop)

    def reset_parameters(self):
        def linear_encoder_init_weights_(m, bias=0.01):
            if isinstance(m, torch.nn.Linear):
                glorot_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(bias)

        self.attrs_linear_encoder.apply(linear_encoder_init_weights_)
        self.attrs_graph_encoder.reset_parameters()
    
    def forward(self, edge_attr, node_attr, edge_index, num_nodes):
        edge_index, num_padding_edges = self.n_ordered_hop(
            edge_index=edge_index,
            num_nodes=num_nodes,
            rm_self_loop=True,
            n_hop=self._n_hop
        )
        
        index_r = edge_index[self._r_edge_idx, :]
        
        edge_attr = torch.cat((
            edge_attr,
            self.padding_edge_attr(
                n=num_padding_edges,
                device=edge_attr.device
            )),
            axis=0
        )
        
        h0 = torch.cat((
            node_attr[index_r],
            edge_attr), axis = 1)

        h1 = self.attrs_linear_encoder(h0)
        
        index_2step = self.two_step_index(index_r)
    
        h2 = self.attrs_graph_encoder(h1, index_2step)
        h3 = self.attrs_pooling_layer(h2, index_r)

        return self.final_btch_norm(h3)
