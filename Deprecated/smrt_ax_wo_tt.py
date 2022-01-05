# %%
import torch
import numpy as np
import json

from ART.DataSplitter import CidRandomSplitter
from ART.Summarizer import EdgeAttrsSummarizer, MolAttrsSummarizer
from ART.Summarizer import NodeKnnSummarizer, NodeAttrsSummarizer
from ART.Evaluater import Evaluater
from ART.ParSet import LinearLayerPars, AttentiveFPPars, PredictorPars
from ART.funcs import doc_to_json_snapshot
from ART.Deprecated.SMRT import SMRT
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from collections import OrderedDict
from math import ceil
from pymongo import MongoClient
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def transform(data: Data) -> Data:
    mol_attr = mol_smry(data, to_one_hot=True, concat=True)
    return Data(
        y=data["mol_attrs"]["rt"],
        SMILES=data["SMILES"],
        formula=data["formula"],
        is_tautomers=data["is_tautomers"],
        scaffold=data["scaffold"],
        n_tt=data["n_tt"],
        wt=mol_attr["wt"],
        volumn=mol_attr["volume"],
        mlogP=mol_attr["mLogP"],
        mol_attr=mol_attr["concat"],
        num_nodes=len(data["node_attrs"]["symbol"]),
        x=torch.cat([
            node_smry(data, to_one_hot=True, concat=True)["concat"],
            knn_smry(data, to_one_hot=True)["token"]
        ], 1),
        edge_index=data["edge_index"],
        edge_attr=edge_smry(data, to_one_hot=True, concat=True)["concat"]
    )


# %%
if __name__ == '__main__':
    # %%
    DATA_ROOT = "/home/cychang/Documents/lc-ms-rt-prediction/Python/SMRT_wo_tt"
    DATA_PROF = "SMRT.json"
    with open("/".join([DATA_ROOT, "raw", DATA_PROF])) as f:
        profile = json.load(f)

    with open(profile["mongo"]["auth_file_path"]) as f:
        login_info = json.load(f)

    BATCH_SIZE = profile["data"]["batch_size"]
    SHUFFLE = profile["data"]["shuffle"]
    SPLITTER = CidRandomSplitter(
        by=profile["data"]["splitter"]["by"],
        frac_train=profile["data"]["splitter"]["frac_train"],
        frac_valid=profile["data"]["splitter"]["frac_valid"],
        frac_test=profile["data"]["splitter"]["frac_test"],
        seed=profile["data"]["splitter"]["seed"]
    )
    REPLICA_NAME = login_info["replicaName"]
    MONGO_USERNAME = login_info["username"]
    MONGO_PASSWORD = login_info["password"]
    MONGO_HOSTS = ",".join(
        [host["host"] + ":" + str(host["port"])
            for host in profile["mongo"]["hosts"]]
    )
    MONGO_AUTH_DB = login_info["authenticationDatabase"]
    MONGO_READ_PREFERENCE = "primary"
    MONGO_CONN_STR = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOSTS}/?authSource={MONGO_AUTH_DB}&replicaSet={REPLICA_NAME}&readPreference={MONGO_READ_PREFERENCE}"
    MONGO_SNAPSHOT_DB = "ax"
    SNAPSHOT_COLLECTION_NAME = profile["mongo"]["collection"]["snapshot"]
    INCLUDE_TT = profile["data"]["include_tautomers"]

    mng_client = MongoClient(MONGO_CONN_STR)
    mng_db = mng_client[MONGO_SNAPSHOT_DB]

    if (not(SNAPSHOT_COLLECTION_NAME in set(mng_db.list_collection_names()))):
        mng_db.create_collection(name=SNAPSHOT_COLLECTION_NAME)
    mngdb_snapshot = mng_db[SNAPSHOT_COLLECTION_NAME]

    EXPERIMENT_NAME = profile["experiment"]["name"]
    mngdb_pipeline = [
        {"$match": {"name": EXPERIMENT_NAME}},
        {"$sort": {"time": -1}},
        {"$limit": 1}
    ]
    ax_snapshot_doc = list(mngdb_snapshot.aggregate(mngdb_pipeline))
    if len(ax_snapshot_doc) == 0:
        AX_SNAPSHOT = None
    else:
        AX_SNAPSHOT = doc_to_json_snapshot(ax_snapshot_doc[0])
        num_completed_trail = len(AX_SNAPSHOT['experiment']['trials'])
        print(f"   - {num_completed_trail} trails have completed.")

    # %%
    print("1. Summarizing training data")
    train_set = SMRT(
        root=DATA_ROOT,
        profile_name="SMRT.json",
        split="train",
        max_num_tautomer=3,
        include_tautomers=INCLUDE_TT
    )
    train_set = train_set.get_all(n_job=10)

    # %%
    print("1.1. Summarizing mol attributes")
    mol_smry = MolAttrsSummarizer(
        data_list=train_set,
        rm_tt=False,
        verbose=True
    )
    # %%
    print("1.2. Summarizing edge attributes")
    edge_smry = EdgeAttrsSummarizer(
        data_list=train_set,
        rm_tt=False,
        verbose=True
    )
    # %%
    print("1.3. Summarizing node attributes")
    node_smry = NodeAttrsSummarizer(
        data_list=train_set,
        rm_tt=False,
        verbose=True
    )
    # %%
    print("1.4. Summarizing knn attributes")
    knn_smry = NodeKnnSummarizer(
        data_list=train_set,
        rm_tt=False,
        n_bin=20,
        discrete_thr=0.01,
        verbose=True
    )
    # <<< Summarizing training data <<<

    # >>> Transform datasets >>>
    # %%
    print("2. Reading data")
    print("2.1. Reading train set")
    train_set_transformed = SMRT(
        root=DATA_ROOT,
        profile_name="SMRT.json",
        split="train",
        max_num_tautomer=3,
        include_tautomers=INCLUDE_TT,
        transform=transform
    )
    # train_set_transformed.get_all(n_job=10)
    # train_set_transformed = [transform(d) for d in train_set_transformed]

    print("2.2. Reading validation set")
    valid_set_transformed = SMRT(
        root=DATA_ROOT,
        profile_name="SMRT.json",
        split="validation",
        max_num_tautomer=3,
        include_tautomers=INCLUDE_TT,
        transform=transform
    )
    # valid_set_transformed = [transform(d) for d in valid_set_transformed]
    # <<< Transform datasets <<<

    # >>> dataset to dataloader >>>
    print("2.3. Dataset to DataLoader")
    train_loader = DataLoader(
        dataset=train_set_transformed,
        shuffle=SHUFFLE,
        batch_size=BATCH_SIZE
    )
    valid_loader = DataLoader(
        dataset=valid_set_transformed,
        shuffle=SHUFFLE,
        batch_size=BATCH_SIZE
    )
    # <<< dataset to dataloader <<<

    # >>> Model architecture parameters >>>
    first_batch = None
    for d_btch in train_loader:
        first_batch = d_btch
        break

    embd_lyr_pars_dict = OrderedDict({
        "mol_attr": LinearLayerPars(
            in_features=int(
                first_batch["mol_attr"].shape[0]/len(first_batch.SMILES)
            ),
            out_features=sum([ceil(
                mol_smry.discrete_vars_smry[var]["entropy"])
                for var in mol_smry.discrete_vars
            ]),
            dropout=(0.1, False),
            batch_norm=True
        ),
        "node_attr": LinearLayerPars(
            in_features=first_batch.num_node_features,
            out_features=sum(
                [ceil(node_smry.discrete_vars_smry[var]["entropy"])
                    for var in node_smry.discrete_vars]
                + [ceil(knn_smry.discrete_vars_smry["token"]["entropy"])
                    * knn_smry.k]
            ),
            dropout=(0.1, False),
            batch_norm=True
        ),
        "edge_attr": LinearLayerPars(
            in_features=first_batch.num_edge_features,
            out_features=sum([
                ceil(edge_smry.discrete_vars_smry[var]["entropy"])
                for var in edge_smry.discrete_vars
            ]),
            dropout=(0.1, False),
            batch_norm=True
        )
    })

    del first_batch, train_set
    # %%
    # >>> Ax server >>>
    print("3. Setting up an Ax server")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"   > Using device: {device}.")
    # if device.type == 'cuda':
    #     print(f"   > Device name: {torch.cuda.get_device_name(0)}")
    #     print('   > Memory Usage:')
    #     print('   > Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    #     print('   > Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    ax_client = AxClient()
    if AX_SNAPSHOT is None:
        ax_client.create_experiment(
            name=EXPERIMENT_NAME,
            parameters=[
                {
                    "name": "learning_rate",
                    "type": "choice",
                    "values": [float(i)/2.0 for i in range(-10, -3)],
                    "value_type": "float",
                    "log_scale": False,
                    "is_ordered": True,
                },
                {
                    "name": "weight_decay",
                    "type": "choice",
                    "values": [-3.0, -2.5, -2.0, -1.5, -1.0],
                    "value_type": "float",
                    "log_scale": False,
                    "is_ordered": True,
                },
                {
                    "name": "afp_num_timesteps",
                    "type": "choice",
                    "values": [1, 2],
                    "value_type": "int",
                    "log_scale": False,
                    "is_ordered": True,
                },
                {
                    "name": "apf_num_layers",
                    "type": "choice",
                    "values": [1, 2, 3],
                    "value_type": "int",
                    "log_scale": False,
                    "is_ordered": True,
                },
                {
                    "name": "afp_hidden_channels",
                    "type": "choice",
                    "values": [32, 64, 128, 256],
                    "value_type": "int",
                    "log_scale": False,
                    "is_ordered": True,
                },
                {
                    "name": "afp_out_channels",
                    "type": "choice",
                    "values": [32, 64, 128],
                    "value_type": "int",
                    "log_scale": False,
                    "is_ordered": True,
                },
                {
                    "name": "num_prdctr_lyr",
                    "type": "choice",
                    "values": [3, 4, 5],
                    "value_type": "int",
                    "log_scale": False,
                    "is_ordered": True,
                },
                {
                    "name": "prdctr_lyr_channels",
                    "type": "choice",
                    "values": list(range(50, 65)),
                    "value_type": "int",
                    "log_scale": False,
                    "is_ordered": True,
                }
            ],
            # objective_name="valid_mse",
            # minimize=True,
            # choose_generation_strategy_kwargs={
            #     "max_parallelism_override": 2
            # },
            objectives={
                "valid_mse": ObjectiveProperties(minimize=True)
            },
            tracking_metric_names=[
                "valid_mse", "epoch", "r", "rho"
            ],
        )
    else:
        ax_client = ax_client.from_json_snapshot(AX_SNAPSHOT)

    # %%
    n_trail = 50
    TQDM_N_COLS = 60
    for _ in range(n_trail):
        parameters, trial_index = ax_client.get_next_trial()
        print(f"\n{EXPERIMENT_NAME} Trail ({trial_index+1:02d}/{n_trail:02d})")

        afp_mdl_pars = AttentiveFPPars(
            in_channels=embd_lyr_pars_dict["node_attr"].out_features,
            hidden_channels=parameters.get("afp_hidden_channels"),
            out_channels=parameters.get("afp_out_channels"),
            edge_dim=embd_lyr_pars_dict["edge_attr"].out_features,
            num_layers=parameters.get("apf_num_layers"),
            num_timesteps=parameters.get("afp_num_timesteps"),
            dropout=0.1
        )

        prdctr_lyr_channels = parameters.get("prdctr_lyr_channels")
        num_prdctr_lyr = parameters.get("num_prdctr_lyr")
        prdctr_lyr_pars = PredictorPars(
            in_features=afp_mdl_pars.out_channels
                + embd_lyr_pars_dict["mol_attr"].out_features
                + 3,
            hidden_features=[prdctr_lyr_channels] * num_prdctr_lyr,
            out_features=1,
            dropout=[0.1] * num_prdctr_lyr + [-1],
            relu=[True] * num_prdctr_lyr + [False],
            batch_norm=False
        )

        evaluater = Evaluater(
            train_loader=train_loader,
            valid_loader=valid_loader,
            learning_rate=parameters.get("learning_rate"),
            embd_lyr_pars_dict=embd_lyr_pars_dict,
            afp_mdl_pars=afp_mdl_pars,
            prdctr_lyr_pars=prdctr_lyr_pars,
            ax_client=ax_client,
            weight_decay=parameters.get("weight_decay"),
            momentum=0,
            device=device,
            max_epoch=100,
            mngdb_snapshot=mngdb_snapshot,
            tqdm_ncols=70
        )

        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data=evaluater.run()
        )

    trials_pars_df = ax_client.get_trials_data_frame()
    trials_pars_df.to_csv(f"./Ax_Results/{EXPERIMENT_NAME}.csv")

    best_parameters, values = ax_client.get_best_parameters()
    print(best_parameters)
    means, covariances = values

    print("\n")
    print("================================= Best Parameters =================================")
    print(f'mean: {means["valid_mse"]:5.3f}, epoch: {int(means["epoch"]):02d}')
    print(best_parameters)
