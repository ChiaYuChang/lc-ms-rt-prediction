# %%
from typing import Optional
from copy import deepcopy
import numpy as np
import torch

from ART.funcs import hop
from torch import nn
from torch._C import device
from torch_geometric import nn as pyg_nn
from torch_geometric.data import Data
from torch.nn.init import xavier_normal_ as glorot_
from torch_geometric.transforms import TwoHop


# %%
def two_hop(edge_index, num_nodes):
    return TwoHop()(Data(edge_index=edge_index, num_nodes=num_nodes)).edge_index

# %%
class AttrsEncoderLayers(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: Optional[int] = None,
            dropout: Optional[float] = 0.0,
            which_edge_index: Optional[str] = "edge_index",
            direction: Optional[str] = "out"):
        
        super().__init__()

        if direction == "out":
            self._r_edge_idx = 0
        elif direction == "in":
            self._r_edge_idx = 1
        else:
            raise ValueError("direction shold be either 'out' or 'in'.")
        
        self._which_edge_index = which_edge_index

        if hidden_channels is None:
            hidden_channels = out_channels
        
        self._n_hops = n_hops
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
                    nn.Dropout(p=dropout, inplace=True)
                )
    
    def graph_encoder(self, in_channels, out_channels, dropout):
        return  pyg_nn.GATConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout=dropout
                )

    def pooling_layer(self):
        return pyg_nn.global_add_pool

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
        edge_index = self.hop(
            edge_index=edge_index,
            num_nodes=num_nodes,
            rm_self_loop=rm_self_loop)
        return edge_index[:, torch.sum(edge_index < len(x), axis=0) == 2]

    def hop(
            self, 
            edge_index: torch.Tensor,
            num_nodes: Optional[int] = None,
            rm_self_loop: Optional[bool] = True
            ) -> torch.Tensor:
       return hop(edge_index, num_nodes, rm_self_loop)
    
    def reset_parameters(self):
        def linear_encoder_init_weights_(m, bias=0.01):
            if isinstance(m, torch.nn.Linear):
                glorot_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(bias)

        self.attrs_linear_encoder.apply(linear_encoder_init_weights_)
        self.attrs_graph_encoder.reset_parameters()
    
    def forward(self, data: Data):
        index_r = data[self._which_edge_index][self._r_edge_idx, :]
        
        h0 = torch.cat((
            data.node_attr[index_r],
            data.edge_attr), axis = 1)

        h1 = self.attrs_linear_encoder(h0)
        
        index_2step = self.two_step_index(index_r)
    
        h2 = self.attrs_graph_encoder(h1, index_2step)
        h3 = self.attrs_pooling_layer(h2, index_r)

        return self.final_btch_norm(h3)

# %%
