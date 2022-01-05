# %%
import pymongo
import json
import pandas as pd
import numpy as np

from joblib import Parallel, delayed
from bson import ObjectId
from pymongo import MongoClient
from itertools import chain

# %%
with open("./SMRT/raw/SMRT.json") as f:
    profile = json.load(f)

with open(profile["mongo"]["auth_file_path"]) as f:
    login_info = json.load(f)

# %%
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
MOLATTRS_COLLECTION_NAME = profile["mongo"]["collection"]["mol_attrs"]
PORCESSED_DATA_COL = profile["mongo"]["collection"]["processed_data"]
database = profile["data"]["database"]
username = profile["data"]["username"]

# %%
N_JOB = 10
train_df = pd.read_csv("./SMRT/processed/train.csv", index_col=0)
train_df._id = train_df._id.map(ObjectId)
train_df_list = np.array_split(train_df, N_JOB)


# %%


# %%
doc_list = Parallel(n_jobs=N_JOB)(delayed(read_docs)(
    df=df,
    conn_str=MONGO_CONN_STR,
    db=MONGO_DB,
    col=PORCESSED_DATA_COL
) for df in train_df_list)

# %%


# %%
idx = 0
obj_id = train_df_list[idx]._id.tolist()

mng_client = MongoClient(MONGO_CONN_STR)
mng_col = mng_client[MONGO_DB][PORCESSED_DATA_COL]
mng_crsr = mng_col.aggregate([{"$match": {"_id": {"$in": obj_id}}}])
list(mng_crsr)

# def process_curser(skip_n, limit_n):



# %%
# Check whether the collection exist
mng_client = MongoClient(MONGO_CONN_STR)
mng_col = mng_client[MONGO_DB][MOLATTRS_COLLECTION_NAME]

# %%
mol_attr_pipeline = [
  {"$match": {"mol_attrs.is_tautomers": False, "atom_attrs.embedding_flag": 0}},
  {"$group": {
      "_id": None,
      "wt_mu": {"$avg": "$mol_attrs.wt"},
      "wt_theta": {"$stdDevSamp": "$mol_attrs.wt"},
      "volume_mu": {"$avg": "$mol_attrs.volume"},
      "volume_theta": {"$stdDevSamp": "$mol_attrs.volume"},
      "mLogP_mu": {"$avg": "$mol_attrs.mLogP"},
      "mLogP_theta": {"$stdDevSamp": "$mol_attrs.mLogP"},
    }},
  {"$project": {
      "_id": 0,
      "wt": {"mu": "$wt_mu", "theta": "$wt_theta"},
      "volume": {"mu": "$volume_mu", "theta": "$volume_theta"},
      "mLogP": {"mu": "$mLogP_mu", "theta": "$mLogP_theta"}
    }}
]
mol_attr_smry = mng_col.aggregate(mol_attr_pipeline).next()

# %%
edge_attr_pipeline = [
  {"$match": {"mol_attrs.is_tautomers": False, "atom_attrs.embedding_flag": 0}},
  {"$project": {"edge_attrs": "$edge_attrs.edge_attrs"}},
  {"$unwind": {"path": "$edge_attrs"}},
  {"$facet": {
      "bondtype": [{"$group": {"_id": "$edge_attrs.bondtype", "count": {"$count": {}}}}],
      "conjugation": [{"$group": {"_id": "$edge_attrs.conjugation", "count": {"$count": {}}}}],
      "in_ring": [{"$group": {"_id": "$edge_attrs.in_ring", "count": {"$count": {}}}}],
      "stereos": [{"$group": {"_id": "$edge_attrs.stereos", "count": {"$count": {}}}}],
      "aromatic": [{"$group": {"_id": "$edge_attrs.aromatic", "count": {"$count": {}}}}]
    }} 
]
edge_attr_smry = mng_col.aggregate(edge_attr_pipeline).next()
for k in edge_attr_smry.keys():
    edge_attr_smry[k] = pd.DataFrame.from_records(edge_attr_smry[k])

# %%
atom_attr_pipeline = [
    {"$match": {"mol_attrs.is_tautomers": False,"atom_attrs.embedding_flag": 0}},
    {"$project": {"atom_attrs": "$atom_attrs.atom_attrs"}},
    {"$unwind": {"path": "$atom_attrs"}},
    {"$facet": {
        "symbol": [{"$group": {"_id": "$atom_attrs.symbol", "count": {"$count": {}}}}],
        "hybridization": [{"$group": {"_id": "$atom_attrs.hybridization", "count": {"$count": {}}}}],
        "degree": [{"$group": {"_id": "$atom_attrs.degree", "count": {"$count": {}}}}],
        "n_hs": [{"$group": {"_id": "$atom_attrs.n_hs", "count": {"$count": {}}}}],
        "formal_charge": [{"$group": {"_id": "$atom_attrs.formal_charge", "count": {"$count": {}}}}],
        "aromaticity": [{"$group": {"_id": "$atom_attrs.aromaticity","count": {"$count": {}}}}],
        "in_ring": [{"$group": {"_id": "$atom_attrs.in_ring", "count": {"$count": {}}}}]}}
]
atom_attr_smry = mng_col.aggregate(edge_attr_pipeline).next()
for k in atom_attr_smry.keys():
    atom_attr_smry[k] = pd.DataFrame.from_records(atom_attr_smry[k])
# %%
num_bin = 20
bins_pipeline = [
    {"$match": {"mol_attrs.is_tautomers": False, "atom_attrs.embedding_flag": 0}},
    {"$project": {"distance": "$atom_attrs.distance"}},
    {"$unwind": {"path": "$distance"}},
    {"$match": {"distance.k": {"$lt": 3}}},
    {"$project": {"distance": "$distance.distance"}},
    {"$group": {
        "_id": None, 
        "distance_max": {"$max": "$distance"},
        "distance_min": {"$min": "$distance"}}},
    {"$addFields": {
        "num_bin": num_bin, "bin_size": {
            "$divide": [{"$subtract": ["$distance_max", "$distance_min"]}, num_bin]
        }}},
    {"$project": {"_id": 0}}
]

bins_smry = mng_col.aggregate(bins_pipeline).next()
bins_lb = np.arange(
    bins_smry["distance_min"],
    bins_smry["distance_max"],
    bins_smry["bin_size"]
)

# %%
