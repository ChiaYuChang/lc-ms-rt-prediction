import torch

from .GraphConvLayer import GraphConvLayer
from ART.ParSet import PredictorPars
from ART.funcs import predictor_par_transform

from collections import OrderedDict
from torch.nn.init import xavier_normal_ as glorot_
from torch_geometric.nn import global_add_pool
from typing import List, Tuple, Union, NamedTuple


class GCNLayerPar(NamedTuple):
    in_channels: int = 128
    out_channels: int = 128
    dropout: Tuple[float, bool] = (0.1, False)
    relu: bool = True
    batch_norm: bool = False


class KensertGCNEncoderPars(NamedTuple):
    in_features: int = 64
    hidden_features : Union[List[int], None] = None
    out_features: int = 1
    dropout: Union[List[float], float] = 0.1
    relu: Union[List[bool], bool] = False
    batch_norm: Union[List[bool], bool] = False


class KensertGCN(torch.nn.Module):
    def __init__(
            self,
            gcn_lyr_pars: Union[List[GCNLayerPar], GCNLayerPar],
            prdctr_lyr_pars: PredictorPars
            ) -> None:
        super().__init__()

        self.graph_embedder = torch.nn.Sequential()
        if isinstance(gcn_lyr_pars, GCNLayerPar):
            gcn_lyr_pars = [gcn_lyr_pars]
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

        prdctr_lyr_pars = predictor_par_transform(
            in_features=prdctr_lyr_pars.in_features,
            hidden_features=prdctr_lyr_pars.hidden_features,
            out_features=prdctr_lyr_pars.out_features,
            dropout=prdctr_lyr_pars.dropout,
            relu=prdctr_lyr_pars.relu,
            batch_norm=prdctr_lyr_pars.batch_norm
        )

        self.predictor = torch.nn.Sequential()
        for i in prdctr_lyr_pars.keys():
            self.prdctr.add_module(
                name="lnr_{}".format(i),
                module=torch.nn.Linear(
                    in_features=prdctr_lyr_pars[i].in_features,
                    out_features=prdctr_lyr_pars[i].out_features
                )
            )
            if prdctr_lyr_pars[i].dropout[0] > 0:
                self.prdctr.add_module(
                    name="drp_{}".format(i),
                    module=torch.nn.Dropout(
                        p=prdctr_lyr_pars[i].dropout[0],
                        inplace=prdctr_lyr_pars[i].dropout[1]
                    )
                )
            if prdctr_lyr_pars[i].relu is True:
                self.prdctr.add_module(
                    name="relu_{}".format(i),
                    module=torch.nn.ReLU()
                )
        self.reset_parameters()

    def reset_parameter(self):
        def init_weights_(m, bias=0.01):
            if isinstance(m, torch.nn.Linear):
                glorot_(m.weight)
                m.bias.data.fill_(bias)

        for layer in self.graph_embedder:
            layer.reset_parameters()

        self.predictor.apply(init_weights_)

    def forward(self, data):
        h0 = data.node_attr
        edge_attr = data.edge_attr
        tilde_A = data.tilde_A

        h1 = self.graph_embedder(tilde_A, h0)
        fp = global_add_pool(h1, data.batch)
        return self.predictor(fp)
