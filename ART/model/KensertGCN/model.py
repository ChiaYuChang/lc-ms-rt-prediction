from turtle import forward
import torch

from .GraphConvLayer import GraphConvLayer
from ART.ParSet import LayerParSet, MultiLayerParSet, LayerParSetType

from typing import Union, Optional
from torch import nn
from torch.nn.init import xavier_normal_ as glorot_
from torch_geometric import nn as pyg_nn


class KensertGCN(nn.Module):
    def __init__(
            self,
            gcn_lyr_pars: Union[LayerParSetType, MultiLayerParSet],
            prdctr_lyr_pars: Union[LayerParSetType, MultiLayerParSet],
            which_tilde_A: Optional[str] = "normalized_adj_matrix",
            which_node_attr: Optional[str] = "node_attr"
            ) -> None:
        super().__init__()
        
        self._which_tilde_A = which_tilde_A
        self._which_node_attr = which_node_attr
        
        if isinstance(gcn_lyr_pars, LayerParSet):
            gcn_lyr_pars = [gcn_lyr_pars]

        if isinstance(gcn_lyr_pars, MultiLayerParSet):
            gcn_lyr_pars = gcn_lyr_pars.unwind()
        
        self.graph_embedder = nn.Sequential()
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
        self.pooling = pyg_nn.global_add_pool

        if isinstance(prdctr_lyr_pars, LayerParSet):
            prdctr_lyr_pars = [prdctr_lyr_pars]

        if isinstance(prdctr_lyr_pars, MultiLayerParSet):
            prdctr_lyr_pars = prdctr_lyr_pars.unwind()

        self.predictor = nn.Sequential()
        for i, par in enumerate(prdctr_lyr_pars):
            fc_lyr = nn.Sequential()
            fc_lyr.add_module(
                name="lnr_{}".format(i),
                module=nn.Linear(
                    in_features=par.in_channels,
                    out_features=par.out_channels,
                    bias=par.bias
                )
            )
            if par.dropout[0] > 0:
                fc_lyr.add_module(
                    name="drp_{}".format(i),
                    module=nn.Dropout(
                        p=par.dropout[0],
                        inplace=par.dropout[1]
                    )
                )
            if par.batch_norm:
                fc_lyr.add_module(
                    name="drp_{}".format(i),
                    module=nn.BatchNorm1d(
                        num_features=par.out_channels
                    )
                )
            if par.relu:
                fc_lyr.add_module(
                    name="relu_{}".format(i),
                    module=nn.ReLU()
                )
            
            self.predictor.add_module(
                name="fc_{}".format(i),
                module=fc_lyr
            )

        self.reset_parameters()

    def reset_parameters(self):
        def embedder_init_weights_(m):
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        self.graph_embedder.apply(embedder_init_weights_)

        def predictor_init_weights_(m, bias=0.01):
            if isinstance(m, nn.Linear):
                glorot_(m.weight)
                m.bias.data.fill_(bias)

        self.predictor.apply(predictor_init_weights_)

    def forward(self, data):
        h0 = data[self._which_node_attr]
        # h0 = data.node_attr
        tilde_A = torch.sparse_coo_tensor(
            data[self._which_tilde_A]["index"],
            data[self._which_tilde_A]["value"],
            (data.num_nodes, data.num_nodes)
        )

        _, h1 = self.graph_embedder((tilde_A, h0))
        fingerprint = self.pooling(h1, data.batch)
        predicted_y = self.predictor(fingerprint)
        return predicted_y


class KensertGCNClassifier(nn.Module):

    def __init__(
            self,
            num_class: int,
            gcn_lyr_pars: Union[LayerParSetType, MultiLayerParSet],
            prdctr_lyr_pars: Union[LayerParSetType, MultiLayerParSet],
            which_tilde_A: Optional[str] = "normalized_adj_matrix",
            which_node_attr: Optional[str] = "node_attr",
            ) -> None:
        
        super().__init__()

        self._kensert_gcn = KensertGCN(
            gcn_lyr_pars=gcn_lyr_pars,
            prdctr_lyr_pars=prdctr_lyr_pars,
            which_tilde_A=which_tilde_A,
            which_node_attr=which_node_attr,
        )
        self._num_class = num_class
        self._softmax = nn.Softmax(dim=1)

    def one_hot(self, x):
        torch.nn.functional.one_hot(x, num_classes=self._num_class)

    def reset_parameters(self):
        self._kensert_gcn.reset_parameters()

    def forward(self, data):
        predicted_y = self._kensert_gcn(data)
        predicted_class_y = self._softmax(torch.mul(predicted_y, data.mw_mask))
        return predicted_class_y