# %%
import torch

from ART.model.KensertGCN.GraphConvLayer import GraphConvLayer
from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, GCNConv
from torch_geometric.data import Data
from torch_geometric import nn as pyg_nn
from torch_geometric.nn import global_add_pool

# %%
def embedder_init_weights_(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()

# %%
gcn_lyr_pars=graph_embedder_parset
if isinstance(gcn_lyr_pars, LayerParSet):
    gcn_lyr_pars = [gcn_lyr_pars]
if isinstance(gcn_lyr_pars, MultiLayerParSet):
    gcn_lyr_pars = gcn_lyr_pars.unwind()

# %%
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
graph_embedder = pyg_nn.Sequential(
    input_args='A, x',
    modules=graph_embedder_modules
)
graph_embedder.apply(embedder_init_weights_)

# %%
data = btch
batch = btch.batch
tt_node_batch = btch.tt_node_batch
tt_graph_batch = btch.tt_graph_batch

# %%
h0 = data["node_attr"]
tilde_A = torch.sparse_coo_tensor(
    data["normalized_adj_matrix"]["index"],
    data["normalized_adj_matrix"]["value"],
    (data.num_nodes, data.num_nodes)
)
h1 = graph_embedder(A=tilde_A, x=h0)

# %%
tt_cov_layr = pyg_nn.GATConv(
    gcn_lyr_pars[-1].in_channels,
    gcn_lyr_pars[-1].in_channels,
    dropout=0.01,
    add_self_loops=False,
    negative_slope=0.01)

tt_edge_index = torch.stack((torch.arange(tt_node_batch.size(0)), tt_node_batch), dim=0)
tt_mol_attr = global_add_pool(h1, tt_node_batch).relu_()
tt_g_embd = tt_cov_layr((h1, tt_mol_attr), tt_edge_index)

# %%
mol_cov_layr = pyg_nn.GATConv(
    gcn_lyr_pars[-1].in_channels,
    gcn_lyr_pars[-1].in_channels,
    dropout=0.01,
    add_self_loops=False,
    negative_slope=0.01)

mol_edge_index = torch.stack((torch.arange(tt_graph_batch.size(0)), tt_graph_batch), dim=0)
mol_attr = global_add_pool(tt_g_embd, tt_graph_batch).relu_()
mol_embd = mol_cov_layr((tt_g_embd, mol_attr), mol_edge_index).relu_()



# %%
from torch.nn import GRUCell
from torch import nn

# %%
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
tt_pooling = pyg_nn.Sequential(
    input_args='h1, tt_edge_index, tt_node_batch',
    modules=[
        (pyg_nn.global_add_pool, "h1, tt_node_batch -> tt_attr"),
        (nn.ReLU(), "tt_attr -> tt_attr"),
        (lambda x1, x2: (x1, x2), 'h1, tt_attr -> x_pair'),
        (tt_attr_cov, 'x_pair, tt_edge_index -> tt_embd'),
        (nn.ELU(), "tt_embd -> tt_embd"),
        (nn.Dropout(p=0.01), "tt_embd -> tt_embd"),
        (tt_gru, "tt_embd, tt_attr -> tt_attr"),
        (nn.ReLU(), "tt_embd -> tt_embd")
    ]
)
# %%
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

mol_pooling = pyg_nn.Sequential(
    input_args='tt_embd, mol_edge_index, tt_graph_batch',
    modules=[
        (pyg_nn.global_add_pool, "tt_embd, tt_graph_batch -> mol_attr"),
        (nn.ReLU(), "mol_attr -> mol_attr"),
        (lambda x1, x2: (x1, x2), 'tt_embd, mol_attr -> x_pair'),
        (mol_attr_cov, 'x_pair, mol_edge_index -> mol_embd'),
        (nn.ELU(), "mol_embd -> mol_embd"),
        (nn.Dropout(p=0.01), "mol_embd -> mol_embd"),
        (mol_gru, "mol_embd, mol_attr -> mol_attr"),
        (nn.ReLU(), "mol_embd -> mol_embd")
    ]
)
# %%
pooling = pyg_nn.Sequential(
    input_args='h1, tt_edge_index, tt_node_batch, mol_edge_index, tt_graph_batch',
    modules=[
        (pyg_nn.global_add_pool, "h1, tt_node_batch -> tt_attr"),
        (nn.ReLU(), "tt_attr -> tt_attr"),
        (lambda x1, x2: (x1, x2), 'h1, tt_attr -> x_pair'),
        (tt_attr_cov, 'x_pair, tt_edge_index -> tt_embd'),
        (nn.ELU(), "tt_embd -> tt_embd"),
        (nn.Dropout(p=0.01), "tt_embd -> tt_embd"),
        (tt_gru, "tt_embd, tt_attr -> tt_attr"),
        (nn.ReLU(), "tt_embd -> tt_embd"),
        (pyg_nn.global_add_pool, "tt_embd, tt_graph_batch -> mol_attr"),
        (nn.ReLU(), "mol_attr -> mol_attr"),
        (lambda x1, x2: (x1, x2), 'tt_embd, mol_attr -> x_pair'),
        (mol_attr_cov, 'x_pair, mol_edge_index -> mol_embd'),
        (nn.ELU(), "mol_embd -> mol_embd"),
        (nn.Dropout(p=0.01), "mol_embd -> mol_embd"),
        (mol_gru, "mol_embd, mol_attr -> mol_attr"),
        (nn.ReLU(), "mol_embd -> mol_embd")
    ]
)
# %%
from ART.model.TTmerNet.model import TTmerNet
model = TTmerNet(
    gcn_lyr_pars=graph_embedder_parset,
    prdctr_lyr_pars=predictor_parset.unwind()
)

# %%
