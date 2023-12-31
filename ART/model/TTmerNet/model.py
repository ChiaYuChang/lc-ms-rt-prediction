import imp
from turtle import forward
import torch
from ART.model.KensertGCN.GraphConvLayer import GraphConvLayer
from ART.ParSet import LayerParSet, MultiLayerParSet, LayerParSetType
from ART.Data import GraphData as Data
from typing import Union, Optional
from torch import dropout, negative, nn
from torch.nn import GRUCell
from torch.nn.init import xavier_normal_ as glorot_
from torch_geometric import nn as pyg_nn

# %%
class TTmerNet(nn.Module):
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
        
        graph_embedder_modules = [None] * len(gcn_lyr_pars)
        for i, par in enumerate(gcn_lyr_pars):
            graph_embedder_modules[i] = (
                GraphConvLayer(
                    in_channels=par.in_channels,
                    out_channels=par.out_channels,
                    dropout=par.dropout,
                    relu=par.relu,
                    batch_norm=par.batch_norm),
                'A, x -> x'
            )
        self.graph_embedder = pyg_nn.Sequential(
            input_args='A, x',
            modules=graph_embedder_modules
        )
        tt_attr_cov = pyg_nn.GATConv(
            gcn_lyr_pars[-1].in_channels,
            gcn_lyr_pars[-1].in_channels,
            dropout=0.01,
            add_self_loops=False,
            negative_slope=0.01
        )
        tt_gru = GRUCell(
            gcn_lyr_pars[-1].in_channels,
            gcn_lyr_pars[-1].in_channels
        )        
        tt_btch_norm_0 = nn.BatchNorm1d(num_features=gcn_lyr_pars[-1].in_channels)
        tt_btch_norm_1 = nn.BatchNorm1d(num_features=gcn_lyr_pars[-1].in_channels)
        tt_btch_norm_2 = nn.BatchNorm1d(num_features=gcn_lyr_pars[-1].in_channels)
        self.tt_pooling = pyg_nn.Sequential(
            input_args='node_embd, tt_edge_index, tt_node_batch',
            modules=[
                (tt_btch_norm_0, "node_embd -> node_embd"),
                (pyg_nn.global_add_pool, "node_embd, tt_node_batch -> tt_graph_mean"),
                (nn.ReLU(), "tt_graph_mean -> tt_graph_mean"),
                (tt_btch_norm_1, "tt_graph_mean -> tt_graph_mean"),
                (lambda x1, x2: (x1, x2), 'node_embd, tt_graph_mean -> x_pair'),
                (tt_attr_cov, 'x_pair, tt_edge_index -> tt_graph_embd'),
                (nn.ELU(), "tt_graph_embd -> tt_graph_embd"),
                (nn.Dropout(p=0.001), "tt_graph_embd -> tt_graph_embd"),
                (tt_gru, "tt_graph_embd, tt_graph_mean -> tt_graph_embd"),
                (nn.ReLU(), "tt_graph_embd -> tt_graph_embd"),
                (tt_btch_norm_2, "tt_graph_embd -> tt_graph_embd"),
            ]
        )
        
        mol_attr_cov = pyg_nn.GATConv(
            gcn_lyr_pars[-1].in_channels,
            gcn_lyr_pars[-1].in_channels,
            dropout=0.01,
            add_self_loops=False,
            negative_slope=0.01
        )
        mol_gru = GRUCell(
            gcn_lyr_pars[-1].in_channels,
            gcn_lyr_pars[-1].in_channels
        )
        mol_btch_norm_0 = nn.BatchNorm1d(num_features=gcn_lyr_pars[-1].in_channels)
        mol_btch_norm_1 = nn.BatchNorm1d(num_features=gcn_lyr_pars[-1].in_channels)
        mol_btch_norm_2 = nn.BatchNorm1d(num_features=gcn_lyr_pars[-1].in_channels)
        self.mol_pooling =pyg_nn.Sequential(
            input_args='tt_graph_embd, mol_edge_index, tt_graph_batch',
            modules=[
                (mol_btch_norm_0, "tt_graph_embd -> tt_graph_embd"),
                (pyg_nn.global_add_pool, "tt_graph_embd, tt_graph_batch -> mol_graph_mean"),
                (nn.ReLU(), "mol_graph_mean -> mol_graph_mean"),
                (mol_btch_norm_1, "mol_graph_mean -> mol_graph_mean"),
                (lambda x1, x2: (x1, x2), 'tt_graph_embd, mol_graph_mean -> x_pair'),
                (mol_attr_cov, 'x_pair, mol_edge_index -> mol_graph_embd'),
                (nn.ELU(), "mol_graph_embd -> mol_graph_embd"),
                (nn.Dropout(p=0.001), "mol_graph_embd -> mol_graph_embd"),
                (mol_gru, "mol_graph_embd, mol_graph_mean -> mol_graph_embd"),
                (nn.ReLU(), "mol_graph_embd -> mol_graph_embd"),
                (mol_btch_norm_2, "mol_graph_embd -> mol_graph_embd")
            ]
        )

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
        def init_weights_(m):
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        self.graph_embedder.apply(init_weights_)
        self.tt_pooling.apply(init_weights_)
        self.mol_pooling.apply(init_weights_)

        def predictor_init_weights_(m, bias=0.01):
            if isinstance(m, nn.Linear):
                glorot_(m.weight)
                m.bias.data.fill_(bias)

        self.predictor.apply(predictor_init_weights_)

    def forward(self, data: Data):
        h0 = data[self._which_node_attr]
        # h0 = data.node_attr
        tilde_A = torch.sparse_coo_tensor(
            data[self._which_tilde_A]["index"],
            data[self._which_tilde_A]["value"],
            (data.num_nodes, data.num_nodes)
        )

        node_embd = self.graph_embedder(A=tilde_A, x=h0)
        tt_edge_index = torch.stack((
            torch.arange(
                data.tt_node_batch.size(0),
                device=data.tt_graph_batch.device
            ), data.tt_node_batch), dim=0)
        
        tt_graph_embd = self.tt_pooling(
            node_embd=node_embd,
            tt_edge_index=tt_edge_index,
            tt_node_batch=data.tt_node_batch)
        
        mol_edge_index = torch.stack((
            torch.arange(
                data.tt_graph_batch.size(0),
                device=data.tt_graph_batch.device
            ), data.tt_graph_batch), dim=0)
        
        mol_graph_embd = self.mol_pooling(
            tt_graph_embd=tt_graph_embd,
            mol_edge_index=mol_edge_index,
            tt_graph_batch=data.tt_graph_batch)

        predicted_y = self.predictor(mol_graph_embd)
        return predicted_y
