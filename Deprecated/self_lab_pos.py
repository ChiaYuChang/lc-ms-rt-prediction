# %%
import gc
import torch
import numpy as np
import json

from ART.DataSplitter import CidRandomSplitter
from ART.Deprecated.RTData import RTData
from ART.Summarizer import EdgeAttrsSummarizer, MolAttrsSummarizer
from ART.Summarizer import NodeKnnSummarizer, NodeAttrsSummarizer
from ART.ARTNet import ARTNet
from ART.ParSet import LinearLayerPars, AttentiveFPPars, PredictorPars
from ART.funcs import doc_to_json_snapshot, json_snapshot_to_doc

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from collections import OrderedDict
from math import ceil
from pymongo import MongoClient
from pymongo.collation import Collation
from scipy.stats import spearmanr, pearsonr
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.profile import get_gpu_memory_from_nvidia_smi
from tqdm import tqdm
from typing import Union


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


def evaluate(
        learning_rate, weight_decay, embd_lyr_pars_dict,
        afp_mdl_pars, prdctr_lyr_pars, device, max_epoch: int = 100,
        mngdb_snapshot: Union[None, Collation] = None
        ):

    print(f"   > Using device: {device}.")
    if device.type == 'cuda':
        print(f"   > Device name: {torch.cuda.get_device_name(0)}")
        free_gpu_mem, used_gpu_mem = get_gpu_memory_from_nvidia_smi()
        gpu_prec = round(used_gpu_mem / free_gpu_mem * 100, 2)
        print(f"GPU Memory: {used_gpu_mem}/{free_gpu_mem} ({gpu_prec}%) megabytes")

    learning_rate = 10**learning_rate
    weight_decay = 10**weight_decay

    model = ARTNet(
        embd_lyr_pars_dict=embd_lyr_pars_dict,
        afp_mdl_pars=afp_mdl_pars,
        prdctr_lyr_pars=prdctr_lyr_pars
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    criterion = torch.nn.MSELoss(reduction='sum')
    epoch = 0
    min_mse = 1e10
    min_std = 1e10
    counter = 0
    train_mse = 1e10
    valid_mse = 1e10

    divider_len = 50
    print("=" * divider_len + " Start " + "=" * divider_len)
    while epoch < max_epoch:
        free_gpu_mem, used_gpu_mem = get_gpu_memory_from_nvidia_smi()
        gpu_prec = round(used_gpu_mem / free_gpu_mem * 100, 2)
        print(f"GPU Memory: {used_gpu_mem}/{free_gpu_mem} ({gpu_prec}) megabytes")
        total_loss = 0.0
        total_examples = 0
        for data in tqdm(
                train_loader, total=len(train_loader),
                desc="    - T", ncols=TQDM_N_COLS):
            data = data.to(device)
            optimizer.zero_grad()
            weight = torch.sqrt(1.0/data.n_tt).double()
            outputs = model(data).squeeze()
            targets = data.y
            # weight loss
            loss = criterion(outputs*weight, targets*weight)
            loss.backward()
            optimizer.step()
            # MSE
            total_loss += criterion(outputs, targets)
            total_examples += data.num_graphs
            torch.cuda.empty_cache()
        train_mse = float(torch.sqrt(total_loss / total_examples).to("cpu"))

        if train_mse is None:
            valid_mse = 1e10
            train_mse = 1e10
            break
        # for validation set
        valid_err = []
        for data in tqdm(
                valid_loader, total=len(valid_loader),
                desc="    - V", ncols=TQDM_N_COLS):
            data = data.to(device)
            outputs = model(data).squeeze()
            targets = data.y
            valid_err += [
                float(
                    (criterion(outputs, targets) / data.num_graphs).to("cpu")
                )
            ]
        gc.collect()
        torch.cuda.empty_cache()
        valid_err = np.sqrt(np.array(valid_err))
        valid_mse = np.mean(valid_err)

        if min_mse >= valid_mse:
            min_mse = valid_mse
            min_std = np.std(valid_err)
            counter = 0
        else:
            if epoch > 25:
                counter += 1
            if counter > 15:
                break

        if epoch > 10 and valid_mse > 300:
            break

        if epoch > 35 and valid_mse > 120:
            break

        epoch += 1
        # if epoch % 10 == 0:
        print(f'> Epoch: {epoch:05d}, Loss (Train): {round(train_mse, 3):5.3f}, Loss (Validation): {round(valid_mse, 3):5.3f}, flag: {counter:02d}')
        gc.collect()

    rt_true = []
    rt_prdt = []
    rt_r = []
    rt_rho = []

    for data in tqdm(
            valid_loader, total=len(valid_loader),
            desc="Summarize Results", ncols=TQDM_N_COLS):
        data = data.to(device)
        rt_true_batch = model(data).squeeze().to("cpu").tolist()
        rt_prdt_batch = data.y.to("cpu").tolist()
        r, _ = spearmanr(rt_true_batch, rt_prdt_batch)
        rho, _ = pearsonr(rt_true_batch, rt_prdt_batch)
        rt_r += [r]
        rt_rho += [rho]
        rt_true += rt_true_batch
        rt_prdt += rt_prdt_batch
    torch.cuda.empty_cache()
    gc.collect()

    rt_true = np.array(rt_true)
    rt_prdt = np.array(rt_prdt)
    r_std = np.std(rt_r)
    rho_std = np.std(rt_rho)
    r, _ = spearmanr(rt_true, rt_prdt)
    rho, _ = pearsonr(rt_true, rt_prdt)

    if mngdb_snapshot is not None:
        json_snapshot = ax_client.to_json_snapshot()
        doc = json_snapshot_to_doc(json_snapshot, compress=True)
        mngdb_snapshot.insert_one(doc)

    print("=" * divider_len + "= End =" + "=" * divider_len)
    return {
        "train_mse": (train_mse, 0),
        "valid_mse": (min_mse, min_std),
        "epoch": (epoch, 0),
        "r": (r, r_std),
        "rho": (rho, rho_std)
    }


# %%
if __name__ == '__main__':
    # %%
    with open("/home/cychang/.mongo_login_info.json") as f:
        login_info = json.load(f)

    # %%
    DATA_ROOT = "./SELF_LAB_POS"
    DATA_PROF = "SELF_LAB_POS.json"
    BATCH_SIZE = 128
    SHUFFLE = True
    SPLITTER = CidRandomSplitter(
        by="tt_id",
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        seed=124124
    )
    REPLICA_NAME = login_info["replicaName"]
    MONGO_USERNAME = login_info["username"]
    MONGO_PASSWORD = login_info["password"]
    MONGO_HOSTS = ",".join(
        [host["host"] + ":" + str(host["port"])
            for host in login_info["hosts"]]
    )
    MONGO_AUTH_DB = login_info["authenticationDatabase"]
    MONGO_READ_PREFERENCE = "primary"
    MONGO_CONN_STR = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOSTS}/?authSource={MONGO_AUTH_DB}&replicaSet={REPLICA_NAME}&readPreference={MONGO_READ_PREFERENCE}"
    MONGO_SNAPSHOT_DB = "ax"
    SNAPSHOT_COLLECTION_NAME = "snapshot"
    mng_client = MongoClient(MONGO_CONN_STR)
    mng_db = mng_client[MONGO_SNAPSHOT_DB]

    if (not(SNAPSHOT_COLLECTION_NAME in set(mng_db.list_collection_names()))):
        mng_db.create_collection(name=SNAPSHOT_COLLECTION_NAME)
    mngdb_snapshot = mng_db[SNAPSHOT_COLLECTION_NAME]

    EXPERIMENT_NAME = "SELF_LAB_POS_B128_Adam"
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
    # >>> Summarizing training data >>>
    print("1. Summarizing training data")
    train_set = RTData(
        root=DATA_ROOT,
        conn_str=MONGO_CONN_STR,
        profile_name=DATA_PROF,
        split="train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        splitter=SPLITTER
    )

    # %%
    print("1.1. Summarizing mol attributes")
    mol_smry = MolAttrsSummarizer(
        data_list=train_set,
        rm_tt=False,
        verbose=True
    )

    print("1.2. Summarizing edge attributes")
    edge_smry = EdgeAttrsSummarizer(
        data_list=train_set,
        rm_tt=False,
        verbose=False
    )
    print("1.3. Summarizing node attributes")
    node_smry = NodeAttrsSummarizer(
        data_list=train_set,
        rm_tt=False,
        verbose=False
    )
    print("1.4. Summarizing knn attributes")
    knn_smry = NodeKnnSummarizer(
        data_list=train_set,
        rm_tt=False,
        n_bin=50,
        discrete_thr=0.01,
        verbose=False
    )
    # <<< Summarizing training data <<<

    # >>> Transform datasets >>>
    print("2. Reading data")
    print("2.1. Reading train set")
    train_set_transformed = RTData(
        root=DATA_ROOT,
        profile_name=DATA_PROF,
        split="train",
        transform=transform,
        pre_transform=None,
        pre_filter=None,
        splitter=SPLITTER
    )
    print("2.2. Reading validation set")
    valid_set_transformed = RTData(
        root=DATA_ROOT,
        profile_name=DATA_PROF,
        split="valid",
        transform=transform,
        pre_transform=None,
        pre_filter=None,
        splitter=SPLITTER
    )
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

    # >>> Ax server >>>
    print("3. Setting up an Ax server")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    TQDM_N_COLS = 70
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

        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data=evaluate(
                learning_rate=parameters.get("learning_rate"),
                weight_decay=parameters.get("weight_decay"),
                embd_lyr_pars_dict=embd_lyr_pars_dict,
                afp_mdl_pars=afp_mdl_pars,
                prdctr_lyr_pars=prdctr_lyr_pars,
                device=device,
                max_epoch=100,
                mngdb_snapshot=mngdb_snapshot
            )
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

    # %%
    # opt_trace = ax_client.get_optimization_trace()
    # render(opt_trace)
