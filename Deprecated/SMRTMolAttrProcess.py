# %%
import itertools
import pickle
import pandas as pd
import numpy as np
import torch
import json
import zlib

from ART.DataSplitter import CidRandomSplitter

from bson import ObjectId
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
from numpy.core.numeric import Inf
from pymongo import MongoClient
from torch_geometric.data import Data
from tqdm.std import tqdm
from typing import Union, Dict


def data_to_doc(
        data: Data,
        compress: bool = True,
        addfields: Union[Dict, None] = None
        ) -> dict:
    binary_data = pickle.dumps(data)
    binary_data_sha256 = sha256(binary_data)
    if compress is True:
        binary_data = zlib.compress(binary_data)
        compressed_by = "zlib"
    else:
        compressed_by = None
    doc = {
        "SMILES": data["SMILES"],
        "InChIKey": data["InChIKey"],
        "tt_id": data["tt_id"],
        "is_tautomers": data["is_tautomers"],
        "rt": data["mol_attrs"]["rt"],
        "time": datetime.utcnow(),
        "sha256": binary_data_sha256.hexdigest(),
        "compress": (compress, compressed_by),
        "binary_data": binary_data
    }
    if addfields is not None:
        doc = doc | addfields
    return doc


def doc_to_data(doc: Dict) -> Data:
    binary_data = doc["binary_data"]
    if doc["compress"][0]:
        if doc["compress"][1] == "zlib":
            binary_data = zlib.decompress(binary_data)
            binary_data_sha256 = sha256(binary_data)
            if binary_data_sha256.hexdigest() != doc["sha256"]:
                return None
    return pickle.loads(binary_data)


def select_by_k_dist(
        distance: pd.DataFrame, k: int = 5, knn_to_vec: bool = False,
        max_distance: float = float(Inf)):
    distance = deepcopy(distance)
    distance = distance[distance.k < k]
    if knn_to_vec is False:
        knn = distance[distance.distance < max_distance]\
            .groupby("from_atom_index")["to_atom_symbol"]\
            .agg([
                ("knn", lambda x: paste(x, sep=" ", fixed=k))
            ])
        knn.reset_index(inplace=True)
        knn = knn.rename(columns={'from_atom_index': 'atom_index'})
        knn.sort_values("atom_index", inplace=True)
        return(knn.knn.tolist())
    else:
        knn = distance\
            .groupby("from_atom_index")[
                ["to_atom_symbol", "distance"]
            ]\
            .agg([(
                "knn", lambda x: list(itertools.chain(x)))
            ])
        knn.reset_index(inplace=True)
        knn.columns = ["atom_index", "knn", "distance"]
        knn.sort_values("atom_index", inplace=True)
        return {"knn": knn.knn.tolist(), "distance": knn.distance.tolist()}


def sample_by_group(obj: pd.DataFrame, size: int, replace: bool = False) -> pd.DataFrame:
    size = min(size, obj.shape[0])
    return(obj.loc[np.random.choice(obj.index, size, replace), :])



# %%
if __name__ == '__main__':
    # %%
    with open("./SMRT/raw/SMRT.json") as f:
        profile = json.load(f)

    with open(profile["mongo"]["auth_file_path"]) as f:
        login_info = json.load(f)

    # set up connection
    REPLICA_NAME = profile["mongo"]["replicaName"]
    MONGO_USERNAME = login_info["username"]
    MONGO_PASSWORD = login_info["password"]
    MONGO_HOSTS = ",".join(
        [host["host"] + ":" + str(host["port"])
            for host in profile["mongo"]["hosts"]]
    )
    MONGO_AUTH_DB = login_info["authenticationDatabase"]
    MONGO_READ_PREFERENCE = "primary"
    MONGO_CONN_STR = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOSTS}/?authSource={MONGO_AUTH_DB}&replicaSet={REPLICA_NAME}&readPreference={MONGO_READ_PREFERENCE}"
    MONGO_DB = profile["mongo"]["database"]
    RT_COLLECTION_NAME = profile["mongo"]["collection"]["rt_data"]
    MOLATTRS_COLLECTION_NAME = profile["mongo"]["collection"]["mol_attrs"]
    PORCESSED_DATA_COL = profile["mongo"]["collection"]["processed_data"]
    database = profile["data"]["database"]
    username = profile["data"]["username"]

    # %%
    # Check whether the collection exist
    with MongoClient(MONGO_CONN_STR) as mng_client:
        mng_db = mng_client[MONGO_DB]
        # Create collection if RT_COLLECTION does not exist
        if (PORCESSED_DATA_COL in set(mng_db.list_collection_names())):
            drop_existed_col = input(f"Drop existed collection {PORCESSED_DATA_COL} [y/N]? ")
            if drop_existed_col.lower() == 'y':
                mng_db.drop_collection(name_or_collection=PORCESSED_DATA_COL)
                mng_db.create_collection(name=PORCESSED_DATA_COL)
        else:
            mng_db.create_collection(name=PORCESSED_DATA_COL)

    # %%
    # Query RT data from 
    rt_id = None
    with MongoClient(MONGO_CONN_STR) as mng_client:
        mng_db = mng_client[MONGO_DB]
        rt_id = list(mng_db[RT_COLLECTION_NAME].find({
            "database": database,
            "username": username
            }, {"id": 1}))

    rt_id = [Data.from_dict({"cid": str(item["_id"])}) for item in rt_id]

    # split data
    print("Split data into training, validation and test set")
    splitter = CidRandomSplitter()
    rt_id_train, rt_id_valid, rt_id_test = splitter(rt_id)
    rt_id_train = set([item["cid"] for item in rt_id_train])
    rt_id_valid = set([item["cid"] for item in rt_id_valid])
    rt_id_test = set([item["cid"] for item in rt_id_test])

    # %%
    with MongoClient(MONGO_CONN_STR) as mng_client:
        mng_db = mng_client[MONGO_DB]

        # query SMRT data
        mol_attr_pipeline = [
            {"$match": {"database": database, "username": username}},
            {"$addFields": {"n_tt": {"$size": "$mol_attrs_id"}}},
            {"$unwind": {"path": "$mol_attrs_id"}},
            {"$lookup": {
                "from": MOLATTRS_COLLECTION_NAME,
                "localField": 'mol_attrs_id',
                "foreignField": '_id',
                "as": 'attrs'}},
            {"$match": {"attrs.atom_attrs.embedding_flag": 0}},
            {"$project": {"rt": 1, "n_tt": 1, "attrs": {"$first": "$attrs"}}},
            {"$project": {
                    "n_tt": 1,
                    "tt_id": "$_id",
                    "num_nodes": "$attrs.mol_attrs.n_node",
                    "mol_attrs": {
                        "rt": "$rt",
                        "wt": "$attrs.mol_attrs.wt",
                        "volume": "$attrs.mol_attrs.volume",
                        "n_hba": "$attrs.mol_attrs.n_hba",
                        "n_hbd": "$attrs.mol_attrs.n_hbd",
                        "n_ring": "$attrs.mol_attrs.n_ring",
                        "n_aromatic_ring": "$attrs.mol_attrs.n_aromatic_ring",
                        "n_aliphatic_ring": "$attrs.mol_attrs.n_aliphatic_ring",
                        "mLogP":  "$attrs.mol_attrs.mLogP"
                    },
                    "InChIKey": "$attrs.mol_attrs.InChIKey",
                    "SMILES": "$attrs.mol_attrs.SMILES",
                    "formula": "$attrs.mol_attrs.formula",
                    "scaffold": "$attrs.mol_attrs.scaffold",
                    "is_tautomers": "$attrs.mol_attrs.is_tautomers",
                    "node_attrs": "$attrs.atom_attrs.atom_attrs",
                    "node_distance": "$attrs.atom_attrs.distance",
                    "edge_attrs": "$attrs.edge_attrs.edge_attrs",
                    "edge_index": "$attrs.edge_attrs.edge_index"}}
        ]
        mng_cursor = mng_db[RT_COLLECTION_NAME].aggregate(mol_attr_pipeline)

        # calculate number of documents
        num_doc = mng_db[RT_COLLECTION_NAME].aggregate([
            {"$match": {"database": database, "username": username}},
            {"$addFields": {"n_tt": {"$size": "$mol_attrs_id"}}},
            {"$unwind": {"path": "$mol_attrs_id"}},
            {"$lookup": {
                    "from": MOLATTRS_COLLECTION_NAME,
                    "localField": 'mol_attrs_id',
                    "foreignField": '_id',
                    "as": 'attrs'
                }},
            {"$match": {"attrs.atom_attrs.embedding_flag": 0}},
            {"$count": "num_doc"}
        ]).next()["num_doc"]

        # insert processed data
        print("Insert data")
        for i, doc in tqdm(enumerate(mng_cursor), total=num_doc):
            data = Data()
            for field in [
                    "n_tt", "tt_id", "SMILES", "InChIKey", "formula", 
                    "is_tautomers", "scaffold", "num_nodes", "mol_attrs"
                    ]:
                if field == "tt_id":
                    data[field] = str(doc[field])
                else:
                    data[field] = doc[field]

            node_attrs = dict()
            indexes = doc["node_attrs"][0].keys()
            for idx in indexes:
                if idx != "index":
                    node_attrs[idx] = [record[idx] for record in doc["node_attrs"]]
            data["node_attrs"] = node_attrs

            data["knn_attrs"] = select_by_k_dist(
                distance=pd.DataFrame.from_records(doc["node_distance"]),
                k=3, knn_to_vec=True)

            edge_attrs = dict()
            indexes = doc["edge_attrs"][0].keys()
            for idx in indexes:
                edge_attrs[idx] = [record[idx] for record in doc["edge_attrs"]]
            data["edge_attrs"] = edge_attrs

            edge_index = [[record["begin"], record["end"]] for record in doc["edge_index"]]
            edge_index = torch.tensor(edge_index).transpose(0, 1)
            data["edge_index"] = edge_index

            group = None
            if data["tt_id"] in rt_id_train:
                group = "train"
            elif data["tt_id"] in rt_id_valid:
                group = "validation"
            elif data["tt_id"] in rt_id_test:
                group = "test"
            else:
                print("error")

            doc = data_to_doc(data, addfields={
                "group": group,
                "database": database,
                "username": username
            })

            insert_result = mng_db.processed_data.insert_one(doc)
            error_counter = 0
            while (insert_result.acknowledged is False) and (error_counter < 10):
                print(f"Insert error, try again... ({error_counter:02d}/10)")
                insert_result = mng_db.processed_data.insert_many(doc)
    print("Done")
