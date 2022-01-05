
from copy import deepcopy
import pickle
import pandas as pd
import numpy as np
import zlib
import json

from hashlib import sha256
from bson import ObjectId
from pymongo import MongoClient
from torch_geometric.data import Dataset, Data
from typing import Callable, Dict, Union
from itertools import chain
from joblib import Parallel, delayed

class SMRT(Dataset):
    def __init__(
            self, root: str,
            profile_name: str = "profile.json",
            pre_filter: Union[Callable, None] = None,
            transform: Union[Callable, None] = None,
            split: str = "train",
            max_num_tautomer: int = 5,
            include_tautomers: bool = True
            ) -> None:
        """
        splitter: A DataSplitter object that use to split data
                  into different sets
        """
        self.root = root
        self.profile_name = profile_name
        self.max_num_tautomer = max_num_tautomer

        with open("/".join([self.root, "raw", self.raw_file_names])) as f:
            self.profile = json.load(f)

        with open(self.profile["mongo"]["auth_file_path"]) as f:
            login_info = json.load(f)

        mngReplicaName = self.profile["mongo"]["replicaName"]
        mngUsername = login_info["username"]
        mngPassword = login_info["password"]
        mngHosts = ",".join(
            [host["host"] + ":" + str(host["port"])
                for host in self.profile["mongo"]["hosts"]]
        )
        mngAuthDB = login_info["authenticationDatabase"]        

        self.split = split
        self._include_tautomers = include_tautomers
        self._mngConnStr = f"mongodb://{mngUsername}:{mngPassword}@{mngHosts}/?authSource={mngAuthDB}&replicaSet={mngReplicaName}&readPreference=primary"
        self._mngDB = self.profile["mongo"]["database"]
        self._mngCol = self.profile["mongo"]["collection"]["processed_data"]
        self._mngClient = MongoClient(self._mngConnStr)
        super().__init__(root, transform=transform, pre_transform=None, pre_filter=pre_filter)

        if split == "train" or split == "training":
            path = self.processed_paths[0]
        elif split == "valid" or split == "validation":
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        else:
            raise ValueError(f"split variables should be \
                either 'train', 'validate', or 'test'")
        self._data_df = pd.read_csv(path, index_col=0)

    @property
    def raw_file_names(self):
        return self.profile_name

    @property
    def processed_file_names(self):
        return ["train.csv", "validation.csv", "test.csv"]

    @property
    def mng_connecting_string(self) -> str:
        self._mgConnStr

    @property
    def include_tautomers(self) -> bool:
        return self._include_tautomers

    def download(self):
        pass

    def _read_data_from_db(self, split: str) -> pd.DataFrame:
        database = self.profile["data"]["database"]
        username = self.profile["data"]["username"]
        non_retained_thr = self.profile["data"]["non_retained_threshold"]["time"]

        if self.profile["data"]["non_retained_threshold"]["unit"] == "min":
            non_retained_thr = non_retained_thr*60

        with MongoClient(self._mngConnStr) as mng_client:
            mng_db = mng_client[self._mngDB]
            mng_col = mng_db[self._mngCol]
            mng_filter = {
                "group": split,
                "database": database,
                "username": username
            }

            if non_retained_thr is not None:
                mng_filter["rt"] = {"$gte": non_retained_thr}

            if self._include_tautomers is False:
                mng_filter["is_tautomers"] = False

            docs_list = list(mng_col.find(
                filter=mng_filter,
                projection={
                    "_id": 1,
                    "SMILES": 1,
                    "InChIKey": 1,
                    "tt_id": 1,
                    "is_tautomers": 1}
            ))

        data_set_df = pd.DataFrame.from_records(docs_list)
        data_set_df._id = data_set_df._id.astype(str)

        def sample_by_group(
            obj: pd.DataFrame, n: int, replace: bool = False
                ) -> pd.DataFrame:
            n = min(n, obj.shape[0])
            return(obj.loc[np.random.choice(obj.index, n, replace), :])

        data_set_df = data_set_df\
            .groupby('tt_id', as_index=True)\
            .apply(func=sample_by_group, n=self.max_num_tautomer, replace=False)\
            .reset_index(drop=True)

        print(f"  > {split} set ({data_set_df.shape[0]})")

        return(data_set_df)

    def process(self):
        for i, split in enumerate(["train", "validation", "test"]):
            data_df = self._read_data_from_db(split=split)
            if self.pre_filter is not None:
                bool_mask = [False] * data_df.shape[0]
                for i, row in data_df.iterrows():
                    bool_mask[i] = self.pre_filter(row)
                data_df = data_df[bool_mask]
            data_df.to_csv(self.processed_paths[i])

    def len(self):
        if self.split == "train":
            return(pd.read_csv(self.processed_paths[0]).shape[0])
        elif self.split == "validation" or self.split == "valid":
            return(pd.read_csv(self.processed_paths[1]).shape[0])
        elif self.split == "test":
            return(pd.read_csv(self.processed_paths[2]).shape[0])
        else:
            ValueError("split should be train, validation or test.")

    def _doc_to_data(self, doc: Dict) -> Data:
        binary_data = doc["binary_data"]
        if doc["compress"][0]:
            if doc["compress"][1] == "zlib":
                binary_data = zlib.decompress(binary_data)
                binary_data_sha256 = sha256(binary_data)
                if binary_data_sha256.hexdigest() != doc["sha256"]:
                    return None
        return pickle.loads(binary_data)

    def get(self, idx):
        mng_col = self._mngClient[self._mngDB][self._mngCol]
        doc = mng_col.find_one({
            "_id": ObjectId(self._data_df.iloc[idx, 0])
        })
        return(self._doc_to_data(doc))

    def get_all(self, n_job: int = 10):
        def read_docs(conn_str, db, col, df, process_func=None):
            obj_ids = df._id.tolist()
            with MongoClient(conn_str) as mng_client:
                mng_col = mng_client[db][col]
                mng_crsr = mng_col.aggregate([{"$match": {"_id": {"$in": obj_ids}}}])
            if process_func is not None:
                return [process_func(doc) for doc in mng_crsr]
            else:
                return list(mng_crsr)

        def doc_to_data(doc: Dict) -> Data:
            binary_data = doc["binary_data"]
            if doc["compress"][0]:
                if doc["compress"][1] == "zlib":
                    binary_data = zlib.decompress(binary_data)
                    binary_data_sha256 = sha256(binary_data)
                    if binary_data_sha256.hexdigest() != doc["sha256"]:
                        return None
            return pickle.loads(binary_data)

        df = deepcopy(self._data_df)
        df._id = df._id.map(ObjectId)
        df_list = np.array_split(df, n_job)

        # print("Read docs from mongoDB.")
        doc_list = Parallel(n_jobs=n_job)(delayed(read_docs)(
            df=df,
            conn_str=self._mngConnStr,
            db=self._mngDB,
            col=self._mngCol,
            process_func=doc_to_data
        ) for df in df_list)

        # print("Parsing docs.")
        # return [self._doc_to_data(doc) for doc in chain.from_iterable(doc_list)]
        return [doc for doc in chain.from_iterable(doc_list)]
