# %%
import json
from os import name
import pandas as pd
import numpy as np
import torch
import pickle

from torch.nn import parameter


from ART.DataSet import SMRT, PredRet
from ART.DataSplitter import RandomSplitter
from ART.Featurizer.FeatureSet import DefaultFeatureSet
from ART.Featurizer.Featurizer import Featurizer, ParallelFeaturizer
from ART.FileReaders import ParallelMolReader
from ART.FileReaders import SMRTSdfReader
from ART.ModelEvaluator import ModelEvaluator
from ART.model.KensertGCN.model import KensertGCN
from ART.ParSet import LayerParSet, LinearLayerParSet, MultiLayerParSet
from ART.ParSet import GCNLayerParSet
from ART.DataTransformer.DataTransformer import DataTransformer
from ART.DataTransformer.Transforms import gen_mw_mask, gen_normalized_adj_matrix
from ART.DataTransformer.Transforms import gen_knn_graph, gen_knn_distance
from ART.DataTransformer.Transforms import gen_radius_graph, gen_radius_distance
from ART.SnapshotSaver import MongoDB
from ART.funcs import check_has_processed, data_to_doc, doc_to_data
from ART.funcs import json_snapshot_to_doc

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from copy import deepcopy
from multiprocessing import Value, cpu_count
from pathos.multiprocessing import ProcessingPool
from pymongo import MongoClient
from rdkit import RDLogger
from time import sleep
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader


# %%
if __name__ == '__main__':
    # %%
    snapshot_db = MongoDB(
        path_to_auth_json="/home/cychang/.mongo_login_info.json"
    )
    snapshot_db.db = {"db": "ax", "col": "snapshot"}
    
    # %%
    root = "./Data/SMRT"
    raw_file_names = "SMRT_dataset.sdf"
    
    # %%
    if not(check_has_processed(
            root=root, raw_file_names=raw_file_names,
            processed_file_names=["pre_filter.pt",  "pre_transform.pt",  "test.pt", 
                                  "train.pt",  "valid.pt",  "smrt_mw.pt"]
        )):
        print(f"Calculating descriptors")
        n_jobs = cpu_count() // 4 * 3
        print(f"Using {n_jobs} cores for preprocessing")
        
        sup_info = {
            "system": "Agilent 1100/1200 series liquid chromatography (LC) system",
            "username": "Xavier Domingo",
            "upload_date": "2019-12-20",
            "rt_unit": "sec"
        }

        RDLogger.DisableLog('rdApp.*')
    
        print("Setting up Reader and Featurizer")
        mol_reader = SMRTSdfReader(
            file_path="/".join((root, "raw", raw_file_names)),
            sup_info = sup_info
        )

        parallel_mol_reader = ParallelMolReader(
            n_jobs=n_jobs,
            inplace=False
        )

        featurizer = Featurizer(
            feature_set=DefaultFeatureSet(),
            include_coordinates=True,
            use_np = True
        )

        parallel_featureizer = ParallelFeaturizer(
            featurizer=featurizer,
            n_jobs=n_jobs,
            rm_None=True
        )

        pool = ProcessingPool(nodes=n_jobs)
        
        print("Reading File")
        smrt = parallel_mol_reader(mol_reader=mol_reader, pool=pool)

        print("Featurizing")
        smrt = parallel_featureizer(file_reader=smrt, pool=pool)

        smrt_df = pd.DataFrame.from_records([{"mw": d.sup["mw"], "rt": d.sup["rt"]} for d in smrt])
        smrt_df.sort_values(["mw", "rt"], inplace=True)
        smrt_df.reset_index(inplace=True)
        smrt_mw = np.array(smrt_df["mw"])
        smrt_y_category = np.array(smrt_df["index"])
        smrt_n = smrt_df.shape[0]
        for i, y_cat in enumerate(smrt_y_category):
            smrt[i]["y_cat"] = np.array([y_cat, smrt_n], dtype=np.compat.long)
            smrt[i]["sup"]["one_hot"] = y_cat
        
        with open("/".join([root, "processed", "smrt_mw.pt"]), "wb") as f:
            pickle.dump(torch.tensor(smrt_mw, dtype=torch.float32), f)
        
        print("Converting np.array to torch.tensor")
        smrt_doc = pool.map(data_to_doc, smrt)
        smrt = pool.map(lambda d: doc_to_data(d,  w_sup_field=True, zero_thr=1.0), smrt_doc)

        print("Setting pre-transform function")
        with open("/".join([root, "processed", "smrt_mw.pt"]), "rb") as f:
            smrt_mw = pickle.load(f)
        gen_mw_mask.args["mw_list"] =  smrt_mw
        pre_transform = DataTransformer(
            transform_list=[
                gen_normalized_adj_matrix,
                gen_mw_mask
            ],
            inplace=True,
            rm_sup_info=True
        )

        print("Setting up InMemoryDataSet")
        smrt = SMRT(
            root=root,
            data_list=smrt,
            pre_transform=pre_transform,
            splitter=RandomSplitter(seed=20211227)
        )
        RDLogger.EnableLog('rdApp.info')

    # %%
    gen_t_A_knn = deepcopy(gen_normalized_adj_matrix)
    gen_t_A_knn.args["which"] = "knn_edge_index"
    gen_t_A_knn.name = "knn_t_A"

    gen_t_A_radius = deepcopy(gen_normalized_adj_matrix)
    gen_t_A_radius.args["which"] = "radius_edge_index"
    gen_t_A_radius.name = "radius_t_A"

    # %%
    # transform = DataTransformer(
    #     transform_list=[
    #         gen_knn_graph,
    #         gen_knn_distance,
    #         gen_t_A_knn,
    #         gen_radius_graph,
    #         gen_radius_distance,
    #         gen_t_A_radius
    #     ],
    #     inplace=True,
    #     rm_sup_info=True
    # )
    transform = None

    # %%
    smrt_tarin = SMRT(
        root=root, 
        split="train",
        transform=transform
    )

    smrt_valid = SMRT(
        root=root, 
        split="valid",
        transform=transform
    )

    smrt_test = SMRT(
        root=root, 
        split="test",
        transform=transform
    )

    # %%
    num_edge_attr = smrt_tarin[0].edge_attr.shape[1]
    num_node_attr = smrt_tarin[0].node_attr.shape[1]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # %%
    ax_exp_name = "202201120143"
    ax_snapshot_id, ax_snapshot = snapshot_db.read_snapshot({"name": ax_exp_name})
    snapshot_db.id = ax_snapshot_id
    ax_client = AxClient()
    
    # %%
    if ax_snapshot is None:
        ax_client.create_experiment(
            name=ax_exp_name,
            parameters=[
                {
                    "name": "batch_size",
                    "type": "choice",
                    "values": [32, 64, 128],
                    "value_type": "int",
                    "log_scale": True
                },
                {
                    "name": "learning_rate",
                    "type": "range",
                    "bounds": [3.0, 4.0],
                    "value_type": "float",
                    "log_scale": True
                },
                {
                    "name": "weight_decay",
                    "type": "range",
                    "bounds": [3.0, 6.0],
                    "value_type": "float",
                    "log_scale": True
                },
                {
                    "name": "num_embedder_layer",
                    "type": "choice",
                    "values": [3, 5],
                    "value_type": "int",
                    "is_ordered": True
                },
                {
                    "name": "embedder_hidden_channels",
                    "type": "range",
                    "bounds": [128, 256],
                    "value_type": "int",
                    "log_scale": False
                },
                                {
                    "name": "embedder_output_channels",
                    "type": "range",
                    "bounds": [128, 512],
                    "value_type": "int",
                    "log_scale": False
                },
                {
                    "name": "num_predictor_layer",
                    "type": "choice",
                    "values": [2, 3],
                    "value_type": "int",
                    "is_ordered": True
                },
                                {
                    "name": "predictor_hidden_channels",
                    "type": "range",
                    "bounds": [256, 1024],
                    "value_type": "int",
                    "log_scale": False
                },
                {
                    "name": "predictor_dropout",
                    "type": "range",
                    "bounds": [0.0, 0.3],
                    "value_type": "float",
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
        ax_client = ax_client.from_json_snapshot(ax_snapshot)
    
    # %%
    num_trail = 50

    for _ in range(num_trail):
        parameters, trial_index = ax_client.get_next_trial()
        print(f"\n{ax_exp_name} Trail ({trial_index+1:02d}/{num_trail:02d})")

        smrt_tarin_loader = DataLoader(
            dataset=smrt_tarin,
            batch_size=parameters["batch_size"],
            shuffle=True
        )

        smrt_valid_loader = DataLoader(
            dataset=smrt_valid,
            batch_size=parameters["batch_size"],
            shuffle=True
        )

        n_e_l = parameters["num_embedder_layer"]
        e_h_c = parameters["embedder_hidden_channels"]
        e_o_c = parameters["embedder_output_channels"]
    
        graph_embedder_parset = MultiLayerParSet(
            in_channels=num_node_attr,
            hidden_channels=[e_h_c] * (n_e_l-1),
            out_channels=e_o_c,
            dropout=(0.1, False),
            relu=True,
            batch_norm=True,
            output_obj=GCNLayerParSet
        )

        n_p_l = parameters["num_predictor_layer"]
        p_h_c = parameters["predictor_hidden_channels"]
        p_d_o  = parameters["predictor_dropout"]
        predictor_parset = MultiLayerParSet(
            in_channels=e_o_c,
            hidden_channels=[p_h_c] * (n_p_l-1),
            out_channels=1,
            dropout=(p_d_o, False),
            relu=[True] * (n_p_l-1) + [False],
            bias=True,
            batch_norm=False,
            output_obj=LinearLayerParSet
        )
    

        model = KensertGCN(
            gcn_lyr_pars=graph_embedder_parset,
            prdctr_lyr_pars=predictor_parset,
            which_tilde_A="normalized_adj_matrix"
        )
    
        optimizer = Adam(
            params=model.parameters(),
            lr=10**(-parameters["learning_rate"]),
            weight_decay=10**(-parameters["weight_decay"]))
        
        # loss = nn.MSELoss()
        loss = nn.HuberLoss()

        evaluator = ModelEvaluator(
            model=model,
            optimizer=optimizer,
            loss=loss,
            train_loader=smrt_tarin_loader,
            valid_loader=smrt_valid_loader,
            device=device,
            max_epoch=300
        )

        ax_trail_result = evaluator.run()
        

        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data=ax_trail_result
        )

        snapshot_doc = ax_client.to_json_snapshot()
        snapshot_db.save_snapshot(snapshot_doc)
        

# %%
# def calc_index(x: torch.Tensor):
#     if x.is_cuda:
#         device = x.device
#         x_cpu = x.to("cpu")
    
#     j = int(max(x_cpu)) + 1
#     idxdict = {}
#     for i in x_cpu.numpy():
#         if i in idxdict:
#             continue
#         else:
#             idxdict[i] = j
#             j += 1

#     return torch.stack(( 
#         x, torch.tensor(np.vectorize(idxdict.get)(x_cpu)).to(device)
#     ))
    

# %%
# import torch_geometric.nn as pyg_nn
# import torch.nn as nn

# encoder = nn.ModuleList([
#     nn.Sequential(
#         nn.BatchNorm1d(
#             num_features=btch.node_attr.shape[1] + btch.edge_attr.shape[1]
#         ),
#         nn.Linear(
#             in_features=btch.node_attr.shape[1] + btch.edge_attr.shape[1],
#             out_features=128,
#             bias=False
#         ),
#         nn.BatchNorm1d(
#             num_features=128
#         ),
#         nn.ReLU(),
#         nn.Dropout(p = 0.0, inplace=False)
#     ),
#     pyg_nn.GATConv(
#         in_channels=128,
#         out_channels=128,
#         dropout=0
#     ),
#     GraphConvLayer(
#         in_channels=128,
#         out_channels=64,
#         dropout=(0.0, False),
#         relu=True,
#         batch_norm=False
#     )
# ])
# encoder.to(device)
# encoder.apply(weight_reset)

# %%
# x_out = torch.cat((
#     btch.node_attr[btch.edge_index[0, :]],
#     btch.edge_attr), axis = 1)

# x_out_edge_index = calc_index(btch.edge_index[0, :])
# h_out_0 = encoder[0](x_out)
# h_out_1 = encoder[1](h_out_0, edge_index=x_out_edge_index)
# h_out_2 = pyg_nn.global_add_pool(h_out_1, x_out_edge_index[0, :])

# tilde_A = torch.sparse_coo_tensor(
#         btch.normalized_adj_matrix["index"],
#         btch.normalized_adj_matrix["value"],
#         (btch.num_nodes, btch.num_nodes)
#     ).to(device)

# _, h_out_3 = encoder[2].forward((tilde_A, h_out_2))

# %%
# x_in = torch.cat((
#     btch.node_attr[btch.edge_index[1, :]],
#     btch.edge_attr), axis = 1)
# x_in_edge_index = calc_index(btch.edge_index[1, :])
# h_in_0 = encoder[0](x_in)
# h_in_1 = encoder[1](h_in_0, edge_index=x_in_edge_index)
# h_in_2 = pyg_nn.global_add_pool(h_in_1, x_in_edge_index[0, :])
# tilde_A = torch.sparse_coo_tensor(
#         btch.normalized_adj_matrix["index"],
#         btch.normalized_adj_matrix["value"],
#         (btch.num_nodes, btch.num_nodes)
#     ).to(device)

# _, h_in_3 = encoder[2].forward((tilde_A, h_in_2))

# %%
