# %%
import torch
import pandas as pd
import numpy as np
import torch
import pickle

from ART.DataSet import SMRT
from ART.DataSplitter import RandomSplitter
from ART.Featurizer.FeatureSet import DefaultFeatureSet
from ART.Featurizer.Featurizer import Featurizer, ParallelFeaturizer
from ART.FileReaders import ParallelMolReader
from ART.FileReaders import SMRTSdfReader
from ART.model.TTmerNet.model import TTmerNet
from ART.model.KensertGCN.GraphConvLayer import GraphConvLayer
from ART.ModelEvaluator import ModelEvaluator, RegressionModelEvaluator
from ART.ParSet import LayerParSet, LinearLayerParSet, MultiLayerParSet
from ART.ParSet import GCNLayerParSet
from ART.DataTransformer.DataTransformer import DataTransformer
from ART.DataTransformer.Transforms import gen_mw_mask, gen_normalized_adj_matrix
from ART.SnapshotSaver import MongoDB
from ART.funcs import check_has_processed, data_to_doc, doc_to_data
from ART.DataTransformer.Transforms import Transform

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from copy import deepcopy
from datetime import datetime
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from rdkit import RDLogger
from scipy.interpolate import interp1d
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, GCNConv
from torch_geometric.data import Data
from torch_geometric import nn as pyg_nn
from torch_geometric.nn import global_add_pool
# %%
snapshot_db = MongoDB(
    path_to_auth_json="/home/cychang/.mongo_login_info.json"
)
snapshot_db.db = {"db": "ax", "col": "snapshot"}

root = "./Data/SMRT"
raw_file_names = "SMRT_dataset.sdf"

df = pd.DataFrame({
    "time": [0, 3, 2, 15, 100],
    "ratio": [3, 3, 50, 85, 85] 
})
df["time"] = np.cumsum(df.time * 60)
f = interp1d(df["time"], df["ratio"])        
def sec2ratio_fun(data, fun):
    return torch.tensor(fun(data.y))

sec2ratio = Transform(
    name="ratio",
    func=sec2ratio_fun,
    args={"fun":f}
)

transform = DataTransformer(
            transform_list=[
                sec2ratio
            ],
            inplace=True,
            rm_sup_info=True
        )

smrt_valid = SMRT(
    root=root, 
    split="valid",
    transform=transform
)

num_edge_attr = smrt_valid[0].edge_attr.shape[1]
num_node_attr = smrt_valid[0].node_attr.shape[1]
num_class = smrt_valid[0].mw_mask.shape[0]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
ax_exp_name = "TTNet_dev"
try:
    ax_snapshot_id, ax_snapshot = snapshot_db.read_snapshot({"name": ax_exp_name})
except ServerSelectionTimeoutError:
    ax_snapshot_id, ax_snapshot = (None, None)
snapshot_db.snapshot_id = ax_snapshot_id
ax_client = AxClient()

if ax_snapshot is None:
    print("Start a new experiment.")
    ax_client.create_experiment(
        name=ax_exp_name,
        parameters=[
            {
                "name": "batch_size",
                "type": "choice",
                "values": [32, 64, 128],
                "value_type": "int",
                "is_ordered": True
            },
            {
                "name": "learning_rate",
                "type": "range",
                "bounds": [4.5, 6.0],
                "value_type": "float",
                "log_scale": True
            },
            {
                "name": "weight_decay",
                "type": "range",
                "bounds": [2.0, 5.0],
                "value_type": "float",
                "log_scale": True
            },
            {
                "name": "num_embedder_layer",
                "type": "range",
                "bounds": [3, 6],
                "value_type": "int",
                "log_scale": False,
            },
            {
                "name": "embedder_hidden_channels",
                "type": "range",
                "bounds": [128, 256],
                "value_type": "int",
                "log_scale": False
            },
            {
                "name": "predictor_hidden_channels",
                "type": "range",
                "bounds": [64, 512],
                "value_type": "int",
                "log_scale": False
            },
            {
                "name": "num_predictor_layer",
                "type": "range",
                "bounds": [3, 6],
                "value_type": "int",
                "log_scale": False
            },
        ],
        objectives={
            "valid_loss": ObjectiveProperties(minimize=True)
        },
        tracking_metric_names=[
            "train_loss", "valid_loss", "epoch", "rmse", "mae", "r", "rho"
        ],
    )
else:
    print("Read snapshot.")
    ax_client = ax_client.from_json_snapshot(ax_snapshot)

num_trail = 50

# %%
parameters, trial_index = ax_client.get_next_trial()
print(f"\n{ax_exp_name} Trail ({trial_index+1:02d}/{num_trail:02d})")

b_s = parameters["batch_size"]

smrt_valid_loader = DataLoader(
    dataset=smrt_valid,
    batch_size=b_s,
    shuffle=True
)

n_e_l = parameters["num_embedder_layer"]
e_h_c = parameters["embedder_hidden_channels"]
e_o_c = parameters["embedder_hidden_channels"]

graph_embedder_parset = MultiLayerParSet(
    in_channels=num_node_attr,
    hidden_channels=[e_h_c] * (n_e_l-1),
    out_channels=e_o_c,
    dropout=(0.1, False),
    relu=True,
    batch_norm=True,
    output_obj=GCNLayerParSet
)

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
btch = next(iter(smrt_valid_loader))
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
node_embd = graph_embedder(A=tilde_A, x=h0)

# %%
# tt_cov_layr = pyg_nn.GATConv(
#     gcn_lyr_pars[-1].in_channels,
#     gcn_lyr_pars[-1].in_channels,
#     dropout=0.01,
#     add_self_loops=False,
#     negative_slope=0.01)
# # tt_btchnrm_lyr = nn.BatchNorm1d(num_features=gcn_lyr_pars[-1].in_channels)
tt_edge_index = torch.stack((torch.arange(tt_node_batch.size(0)), tt_node_batch), dim=0)
# tt_graph_embd = global_add_pool(node_embd, tt_node_batch).relu_()
# # tt_graph_embd = tt_btchnrm_lyr(tt_graph_embd)
# tt_graph_embd = tt_cov_layr((node_embd, tt_graph_embd), tt_edge_index)



# %%
# mol_cov_layr = pyg_nn.GATConv(
#     gcn_lyr_pars[-1].in_channels,
#     gcn_lyr_pars[-1].in_channels,
#     dropout=0.01,
#     add_self_loops=False,
#     negative_slope=0.01)

mol_edge_index = torch.stack((torch.arange(tt_graph_batch.size(0)), tt_graph_batch), dim=0)
# mol_graph_embd = global_add_pool(tt_graph_embd, tt_graph_batch).relu_()
# mol_embd = mol_cov_layr((tt_graph_embd, tt_graph_embd), mol_edge_index).relu_()



# %%
from torch.nn import GRUCell
from torch import nn

# %%
tt_btch_norm_0 = nn.BatchNorm1d(num_features=gcn_lyr_pars[-1].in_channels)
tt_btch_norm_1 = nn.BatchNorm1d(num_features=gcn_lyr_pars[-1].in_channels)
tt_btch_norm_2 = nn.BatchNorm1d(num_features=gcn_lyr_pars[-1].in_channels)
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
tt_graph_embd = tt_pooling(node_embd, tt_edge_index, tt_node_batch)
# %%
mol_btch_norm_0 = nn.BatchNorm1d(num_features=gcn_lyr_pars[-1].in_channels)
mol_btch_norm_1 = nn.BatchNorm1d(num_features=gcn_lyr_pars[-1].in_channels)
mol_btch_norm_2 = nn.BatchNorm1d(num_features=gcn_lyr_pars[-1].in_channels)
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
