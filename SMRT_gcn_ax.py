# %%
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
from ART.ModelEvaluator import ModelEvaluator
from ART.model.KensertGCN.model import KensertGCN
from ART.ParSet import LayerParSet, LinearLayerParSet, MultiLayerParSet
from ART.ParSet import GCNLayerParSet
from ART.DataTransformer.DataTransformer import DataTransformer
from ART.DataTransformer.Transforms import gen_normalized_adj_matrix
from ART.DataTransformer.Transforms import gen_mw_mask, gen_mw_ppm
# from ART.DataTransformer.Transforms import gen_knn_graph, gen_knn_distance
# from ART.DataTransformer.Transforms import gen_radius_graph, gen_radius_distance
from ART.SnapshotSaver import MongoDB
from ART.funcs import check_has_processed, data_to_doc, doc_to_data
from ART.DataTransformer.Transforms import gen_mw_mask

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from copy import deepcopy
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool
from rdkit import RDLogger
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
    # gen_t_A_knn = deepcopy(gen_normalized_adj_matrix)
    # gen_t_A_knn.args["which"] = "knn_edge_index"
    # gen_t_A_knn.name = "knn_t_A"

    # gen_t_A_radius = deepcopy(gen_normalized_adj_matrix)
    # gen_t_A_radius.args["which"] = "radius_edge_index"
    # gen_t_A_radius.name = "radius_t_A"

    # %%
    # gen_mw_mask.args["thr"] = 20
    # transform = DataTransformer(
        # transform_list=[
            # gen_mw_mask
            # gen_knn_graph,
            # gen_knn_distance,
            # gen_t_A_knn,
            # gen_radius_graph,
            # gen_radius_distance,
            # gen_t_A_radius
        # ],
        # inplace=True,
        # rm_sup_info=True
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
    ax_exp_name = "202201251245"
    ax_snapshot_id, ax_snapshot = snapshot_db.read_snapshot({"name": ax_exp_name})
    snapshot_db.id = ax_snapshot_id
    ax_client = AxClient()
    
    # %%
    if ax_snapshot is None:
        ax_client.create_experiment(
            name=ax_exp_name,
            parameters=[
                {
                    "name": "learning_rate",
                    "type": "range",
                    "bounds": [3.3, 3.6],
                    "value_type": "float",
                    "log_scale": True
                },
                {
                    "name": "weight_decay",
                    "type": "range",
                    "bounds": [3.6, 4.1],
                    "value_type": "float",
                    "log_scale": True
                },
                {
                    "name": "num_embedder_layer",
                    "type": "range",
                    "bounds": [5, 6],
                    "value_type": "int",
                    "log_scale": False,
                },
                {
                    "name": "embedder_hidden_channels",
                    "type": "range",
                    "bounds": [180, 205],
                    "value_type": "int",
                    "log_scale": False
                },
                {
                    "name": "predictor_hidden_channels",
                    "type": "range",
                    "bounds": [550, 700],
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
        ax_client = ax_client.from_json_snapshot(ax_snapshot)
    
    # %%
    num_trail = 50

    for _ in range(num_trail):
        # %%
        parameters, trial_index = ax_client.get_next_trial()
        print(f"\n{ax_exp_name} Trail ({trial_index+1:02d}/{num_trail:02d})")

        # b_s = parameters["batch_size"]
        b_s = 32
        smrt_tarin_loader = DataLoader(
            dataset=smrt_tarin,
            batch_size=b_s,
            shuffle=True
        )

        smrt_valid_loader = DataLoader(
            dataset=smrt_valid,
            batch_size=b_s,
            shuffle=True
        )

        n_e_l = parameters["num_embedder_layer"]
        e_h_c = parameters["embedder_hidden_channels"]
        # e_o_c = parameters["embedder_output_channels"] # output layer channels == hidden layer channels
        e_o_c = parameters["embedder_hidden_channels"]
        
        graph_embedder_parset = MultiLayerParSet(
            in_channels=num_node_attr,
            hidden_channels=[e_h_c] * (n_e_l-1),
            out_channels=e_o_c,
            dropout=(0.0, False),
            relu=True,
            batch_norm=True,
            output_obj=GCNLayerParSet
        )

        # n_p_l = parameters["num_predictor_layer"]
        n_p_l = 3
        p_h_c = parameters["predictor_hidden_channels"]
        # p_d_o  = parameters["predictor_dropout"]
        p_d_o = 0.01
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
        # %%
        optimizer = Adam(
            params=model.parameters(),
            lr=10**(-parameters["learning_rate"]),
            weight_decay=10**(-parameters["weight_decay"]))
        
        # loss = nn.MSELoss()
        loss = nn.HuberLoss(reduction='mean', delta=1.0)

        evaluator = ModelEvaluator(
            model=model,
            optimizer=optimizer,
            loss=loss,
            train_loader=smrt_tarin_loader,
            valid_loader=smrt_valid_loader,
            device=device,
            max_epoch=300
        )
        
        ax_trail_result = evaluator.run(trial_index=trial_index)
        

        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data=ax_trail_result
        )

        snapshot_doc = ax_client.to_json_snapshot()
        snapshot_db.save_snapshot(snapshot_doc)
        


