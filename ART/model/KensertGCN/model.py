import torch

from .GraphConvLayer import GraphConvLayer
from ART.ParSet import LayerParSet, MultiLayerParSet, LayerParSetType

from torch.nn.init import xavier_normal_ as glorot_
from torch_geometric.nn import global_add_pool

class KensertGCN(torch.nn.Module):
    def __init__(
            self,
            gcn_lyr_pars: LayerParSetType,
            prdctr_lyr_pars: LayerParSetType
            ) -> None:
        super().__init__()

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
            self.prdctr.add_module(
                name="lnr_{}".format(i),
                module=torch.nn.Linear(
                    in_features=par.in_features,
                    out_features=par.out_features
                )
            )
            if par.dropout[0] > 0:
                self.prdctr.add_module(
                    name="drp_{}".format(i),
                    module=torch.nn.Dropout(
                        p=par.dropout[0],
                        inplace=par.dropout[1]
                    )
                )
            if par.batch_norm:
                self.prdctr.add_module(
                    name="drp_{}".format(i),
                    module=torch.nn.BatchNorm1d(
                        num_features=par.out_features
                    )
                )
            if par.relu:
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
        tilde_A = torch.sparse_coo_tensor(
            data.normalized_adj_matrix["index"],
            data.normalized_adj_matrix["value"],
            (data.num_nodes, data.num_nodes)
        )

        h1 = self.graph_embedder(tilde_A, h0)
        fp = self.pooling(h1, data.batch)
        return self.predictor(fp)
