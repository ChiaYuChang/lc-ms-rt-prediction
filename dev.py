# %%
import json
import pandas as pd
import numpy as np
import torch
import pickle

from ART.DataSet import SMRT, PredRet
from ART.DataSplitter import RandomSplitter
from ART.Featurizer.FeatureSet import DefaultFeatureSet
from ART.Featurizer.Featurizer import Featurizer, ParallelFeaturizer
from ART.FileReaders import ParallelMolReader
from ART.FileReaders import SMRTSdfReader
from ART.model.KensertGCN.GraphConvLayer import GraphConvLayer
from ART.model.KensertGCN.model import KensertGCN
from ART.ParSet import LayerParSet, MultiLayerParSet
from ART.DataTransformer.DataTransformer import DataTransformer
from ART.DataTransformer.Transforms import gen_mw_mask, gen_normalized_adj_matrix
from ART.DataTransformer.Transforms import gen_knn_graph, gen_knn_distance
from ART.DataTransformer.Transforms import gen_radius_graph, gen_radius_distance
from ART.funcs import check_has_processed, data_to_doc, doc_to_data
from ART.funcs import data_to_doc
from copy import deepcopy
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool
from rdkit import RDLogger
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# %%
if __name__ == '__main__':
    # %%
    with open("/home/cychang/.mongo_login_info.json") as f:
        db_auth = json.load(f)
        db_auth["host"] = ",".join(
            [host["host"] + ":" + str(host["port"])
                for host in db_auth["hosts"]]
        )
        db_conn_str = f'mongodb://{db_auth["username"]}:{db_auth["password"]}@{db_auth["host"]}/?authSource={db_auth["authenticationDatabase"]}&replicaSet={db_auth["replicaName"]}&readPreference=primary'
    db = {
        "snapshot": {"db": "ax", "col": "snapshot"},
        "data": {"db": "mols", "col": "SMRTProcessedData"}
    }

    # if (not(SNAPSHOT_COLLECTION_NAME in set(mng_db.list_collection_names()))):
    #     mng_db.create_collection(name=SNAPSHOT_COLLECTION_NAME)
    #     mngdb_snapshot = mng_db[SNAPSHOT_COLLECTION_NAME]

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
        for i, y_cat in enumerate(smrt_y_category):
            smrt[i]["y_cat"] = [y_cat]
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

        # %%
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
    transform = DataTransformer(
        transform_list=[
            gen_knn_graph,
            gen_knn_distance,
            gen_t_A_knn,
            gen_radius_graph,
            gen_radius_distance,
            gen_t_A_radius
        ],
        inplace=True,
        rm_sup_info=True
    )

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
    smrt_tarin_loader = DataLoader(
        dataset=smrt_tarin,
        batch_size=512,
        shuffle=True
    )

    smrt_valid_loader = DataLoader(
        dataset=smrt_valid,
        batch_size=512,
        shuffle=True
    )

    smrt_valid_test = DataLoader(
        dataset=smrt_test,
        batch_size=512,
        shuffle=True
    )

    # %%
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # btch = next(iter(smrt_tarin_loader)).to(device)
    
    # gcn_par = LayerParSet(
    #     in_channels=btch.node_attr.shape[1],
    #     out_channels=64,
    #     dropout=(0.1, True),
    #     relu=True,
    #     batch_norm=True        
    # )

    # gcn_lyr = GraphConvLayer(
    #     in_channels=gcn_par.in_channels,
    #     out_channels=gcn_par.out_channels,
    #     dropout=gcn_par.dropout,
    #     relu=gcn_par.relu,
    #     batch_norm=gcn_par.batch_norm
    # ).to(device)
    # gcn_lyr.reset_parameters()

    # %%
    # tilde_A = torch.sparse_coo_tensor(
    #         btch.normalized_adj_matrix["index"],
    #         btch.normalized_adj_matrix["value"],
    #         (btch.num_nodes, btch.num_nodes)
    #     ).to(device)
    
    # gcn_lyr.forward(tilde_A, btch.node_attr)
