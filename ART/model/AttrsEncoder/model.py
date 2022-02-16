import imp
import torch

from ART.model.AttrsEncoder.AttrsEncoderLayers import AttrsEncoderLayers
from ART.model.KensertGCN.GraphConvLayer import GraphConvLayer
from ART.ParSet import LayerParSet, MultiLayerParSet, LayerParSetType, AttrsEncoderPar

from typing import Optional, Union
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool
from torch.nn.init import xavier_normal_ as glorot_

class ARTAttrEncoderGCN(nn.Module):
    def __init__(
            self,
            attr_emdb_lyr_pars: AttrsEncoderPar,
            gcn_lyr_pars: Union[LayerParSetType, MultiLayerParSet],
            prdctr_lyr_pars: Union[LayerParSetType, MultiLayerParSet],
            which_node_attr: Optional[str] = "node_attr",
            which_edge_attr: Optional[str] = "egde_attr",
            which_edge_index: Optional[str] = "edge_index",
            which_tilde_A: Optional[str] = "normalized_adj_matrix"
            ):
        super().__init__()
        
        self.attr_embedder = AttrsEncoderLayers(
            num_node_attr = attr_emdb_lyr_pars.num_node_attr,
            num_edge_attr = attr_emdb_lyr_pars.num_edge_attr,
            out_channels = attr_emdb_lyr_pars.out_channels,
            hidden_channels = attr_emdb_lyr_pars.hidden_channels,
            n_hop = attr_emdb_lyr_pars.n_hop,
            dropout = attr_emdb_lyr_pars.dropout,
            direction = attr_emdb_lyr_pars.direction
        )

        self._which_tilde_A = which_tilde_A
        self._which_node_attr = which_node_attr
        self._which_edge_attr = which_edge_attr
        self._which_edge_index = which_edge_index

        if isinstance(gcn_lyr_pars, LayerParSet):
            gcn_lyr_pars = [gcn_lyr_pars]

        if isinstance(gcn_lyr_pars, MultiLayerParSet):
            gcn_lyr_pars = gcn_lyr_pars.unwind()
        
        self.graph_embedder = torch.nn.Sequential()
        for i, par in enumerate(gcn_lyr_pars):
            self.graph_embedder.add_module(
                name="gcn_{}".format(i),
                module=GraphConvLayer(
                    in_channels=par.in_channels,
                    out_channels=par.out_channels,
                    dropout=par.dropout,
                    relu=par.relu,
                    batch_norm=par.batch_norm
                )
            )
        self.pooling = global_add_pool

        if isinstance(prdctr_lyr_pars, LayerParSet):
            prdctr_lyr_pars = [prdctr_lyr_pars]

        if isinstance(prdctr_lyr_pars, MultiLayerParSet):
            prdctr_lyr_pars = prdctr_lyr_pars.unwind()

        self.predictor = torch.nn.Sequential()
        for i, par in enumerate(prdctr_lyr_pars):
            fc_lyr = torch.nn.Sequential()
            fc_lyr.add_module(
                name="lnr_{}".format(i),
                module=torch.nn.Linear(
                    in_features=par.in_channels,
                    out_features=par.out_channels,
                    bias=par.bias
                )
            )
            if par.dropout[0] > 0:
                fc_lyr.add_module(
                    name="drp_{}".format(i),
                    module=torch.nn.Dropout(
                        p=par.dropout[0],
                        inplace=par.dropout[1]
                    )
                )
            if par.batch_norm:
                fc_lyr.add_module(
                    name="drp_{}".format(i),
                    module=torch.nn.BatchNorm1d(
                        num_features=par.out_channels
                    )
                )
            if par.relu:
                fc_lyr.add_module(
                    name="relu_{}".format(i),
                    module=torch.nn.ReLU()
                )
            
            self.predictor.add_module(
                name="fc_{}".format(i),
                module=fc_lyr
            )
        
        self.reset_parameters()

    def reset_parameters(self):
        self.attr_embedder.reset_parameters()
        
        def embedder_init_weights_(m):
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        self.graph_embedder.apply(embedder_init_weights_)

        def predictor_init_weights_(m, bias=0.01):
            if isinstance(m, torch.nn.Linear):
                glorot_(m.weight)
                m.bias.data.fill_(bias)

        self.predictor.apply(predictor_init_weights_)
    
    def forward(self, data: Data):
        mixed_attr = self.attr_embedder(
            edge_attr=data[self._which_edge_attr],
            node_attr=data[self._which_node_attr],
            edge_index=data[self._which_edge_index],
            num_nodes=data.num_nodes
        )

        tilde_A = torch.sparse_coo_tensor(
            data[self._which_tilde_A]["index"],
            data[self._which_tilde_A]["value"],
            (data.num_nodes, data.num_nodes)
        )

        _, h1 = self.graph_embedder((tilde_A, mixed_attr))
        fingerprint = self.pooling(h1, data.batch)
        predicted_y = self.predictor(fingerprint)
        return predicted_y
