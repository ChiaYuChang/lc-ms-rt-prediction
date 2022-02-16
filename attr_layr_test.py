# %%
import pickle
import pandas as pd
import numpy as np
import torch

from ART.DataSet import SMRT
# from ART.DataTransformer.Transforms import gen_knn_graph, gen_knn_distance
# from ART.DataTransformer.Transforms import gen_radius_graph, gen_radius_distance
from ART.ModelEvaluator import ModelEvaluator
from ART.model.AttrsEncoder.model import ARTAttrEncoderGCN
from ART.ParSet import AttrsEncoderPar, MultiLayerParSet, LinearLayerParSet
from ART.ParSet import LinearLayerParSet, MultiLayerParSet
from ART.ParSet import GCNLayerParSet
from ART.SnapshotSaver import MongoDB


from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
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
    transform = None
    
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

    # %%
    num_edge_attr = smrt_tarin[0].edge_attr.shape[1]
    num_node_attr = smrt_tarin[0].node_attr.shape[1]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # %%
    ax_exp_name = "AttrEncoderLyrTest_20220118"
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
                    "is_ordered": True
                },
                {
                    "name": "learning_rate",
                    "type": "range",
                    "bounds": [3.5, 5.0],
                    "value_type": "float",
                    "log_scale": True
                },
                {
                    "name": "weight_decay",
                    "type": "range",
                    "bounds": [3.0, 7.0],
                    "value_type": "float",
                    "log_scale": True
                },
                {
                    "name": "attr_encoder_hidden_channels",
                    "type": "range",
                    "bounds": [128, 256],
                    "value_type": "int",
                    "log_scale": False
                },
                {
                    "name": "attr_encoder_n_hop",
                    "type": "range",
                    "bounds": [1, 3],
                    "value_type": "int",
                    "log_scale": False
                },
                {
                    "name": "num_embedder_layer",
                    "type": "range",
                    "bounds": [3, 5],
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
                    "bounds": [0.1, 0.3],
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
    # %%
    for _ in range(num_trail):
        # %%
        parameters, trial_index = ax_client.get_next_trial()
        print(f"\n{ax_exp_name} Trail ({trial_index+1:02d}/{num_trail:02d})")

        # %%
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

        a_e_h_c = parameters["attr_encoder_hidden_channels"]
        a_e_o_c = a_e_h_c
        a_e_n = parameters["attr_encoder_n_hop"]
        
        graph_attr_encoder_parset = AttrsEncoderPar(
            num_node_attr=num_node_attr,
            num_edge_attr=num_edge_attr,
            out_channels=a_e_o_c,
            hidden_channels=a_e_h_c,
            n_hop=a_e_n,
            dropout = (0.1, False),
            which_edge_index="edge_index",
            direction="out"
        )

        n_e_l = parameters["num_embedder_layer"]
        e_h_c = parameters["embedder_hidden_channels"]
        # e_o_c = parameters["embedder_output_channels"] # output layer channels == hidden layer channels
        e_o_c = parameters["embedder_hidden_channels"]

        graph_embedder_parset = MultiLayerParSet(
            in_channels=a_e_o_c,
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

        # %%
        model = ARTAttrEncoderGCN(
            attr_emdb_lyr_pars=graph_attr_encoder_parset,
            gcn_lyr_pars=graph_embedder_parset,
            prdctr_lyr_pars=predictor_parset,
            which_node_attr="node_attr",
            which_edge_attr="edge_attr",
            which_edge_index="edge_index",
            which_tilde_A="normalized_adj_matrix"
        )

        # %%
        optimizer = Adam(
            params=model.parameters(),
            lr=10**(-parameters["learning_rate"]),
            weight_decay=10**(-parameters["weight_decay"]))
        
        # loss = nn.MSELoss()
        loss = nn.HuberLoss(reduction='mean', delta=1.0)

        # %%
        evaluator = ModelEvaluator(
            model=model,
            optimizer=optimizer,
            loss=loss,
            train_loader=smrt_tarin_loader,
            valid_loader=smrt_valid_loader,
            device=device,
            max_epoch=350,
            count_down_thr=75
        )
        
        ax_trail_result = evaluator.run(trial_index=trial_index)
        

        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data=ax_trail_result
        )

        snapshot_doc = ax_client.to_json_snapshot()
        snapshot_db.save_snapshot(snapshot_doc)

