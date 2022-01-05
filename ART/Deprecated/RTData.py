import collections
import itertools
import pickle
import numpy as np
import pandas as pd
import pymongo
import torch
import json

from sys import version_info
from copy import deepcopy
from numpy.core.numeric import Inf
from pandas.core.frame import DataFrame
from random import sample
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.hetero_data import NodeOrEdgeStorage
from tqdm.std import tqdm
from typing import Callable, Union, List
from re import sub as str_sub

from ..DataSplitter import RandomSplitter, DataSplitter


class MongodReader:
    def __init__(
            self,
            conn_str: Union[str, None] = None,
            host: Union[str, None] = "mongodb://localhost",
            port: Union[str, None] = 27017,
            db: str = "mols",
            col_rt: str = "RT_data",
            col_attr: str = "MolAttrs") -> None:

        if conn_str is None:
            self._mng_client = pymongo.MongoClient(host=host, port=port)
        else:
            self._mng_client = pymongo.MongoClient(host=conn_str)

        self._mng_db = self._mng_client[db]
        self._mng_col_rt = self._mng_db[col_rt]
        self._mng_col_attr = self._mng_db[col_attr]
        self._rt_data = None
        self._rt_query = None
        self._data_summary = {}
        self._attr_query = None
        self._data_list = None
        self._knn_to_vec = False
        self._max_distance = float(Inf)
        self._k = None

    @property
    def rt_data(self) -> Union[DataFrame, None]:
        return self._rt_data

    @property
    def data_list(self) -> Union[List[Data], None]:
        return self._data_list

    @property
    def rt_query(self) -> Union[dict, None]:
        return self._rt_query

    @property
    def k(self):
        return self._k

    @property
    def max_distance(self) -> Union[float, None]:
        return self._max_distance

    @property
    def data_summary(self) -> dict:
        return self._data_summary

    def read_rt(
            self, query: Union[dict, None],
            exclude_cols: Union[List[str], None] = None) -> None:
        """Read retention time (RT) data from Mongodb"""
        # For object initialization
        if exclude_cols is None:
            exclude_cols = [
                "SMILE", "InChI", "unit", "dataset",
                "database", "username", "upload_date",
                "name", "PubChemCid"
            ]

        total_rt_data = self._mng_col_rt.count_documents(query)
        if total_rt_data < 1:
            print("Cannot query any document from the database.")
        else:
            rt_data = self._mng_col_rt.find(query)
            self._rt_query = query
            data_rt = pd.DataFrame\
                .from_records(rt_data, exclude=exclude_cols)\
                .rename(columns={"_id": "record_id"})
            data_rt["record_id"] = data_rt.record_id.astype(str)
            data_rt = data_rt.set_index("record_id")
            self._rt_data = data_rt
        return None

    def read_mol_attrs(
            self, query: Union[None, dict] = None,
            n_mol: Union[int, None] = None,
            k: int = 3,
            max_distance: float = None,
            knn_to_vec: Union[None, bool] = None,
            add_cid_by_tt: bool = True,
            limit: Union[int, None] = None):
        """Read molecular info from Mongodb"""

        def paste(x, sep: str = "", fixed: int = None):
            if fixed is None:
                return(sep.join(x))
            else:
                padding_len = (fixed - len(x))
                if padding_len < 0:
                    raise ValueError("The fixed paramenter should be greater \
                        or equal to the length of x.")
                else:
                    padding_strs = ["_"] * padding_len
                x = x.tolist() + padding_strs
                return(sep.join(x))

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
                return((knn.knn.tolist(), knn.distance.tolist()))

        def gen_data(mol_attr, data_rt_idx, add_cid_by_tt) -> None:
            """Construct data objects from dictionaries"""
            # initialize an empty object
            data = Data()
            data["mol_attrs"] = {}
            data["node_attrs"] = {}
            data["edge_attrs"] = {}
            data["knn_attrs"] = {}
            
            # Ids
            rt_doc_ids = mol_attr["rt_data_id"]
            rt_doc_id = list(set(rt_doc_ids) & data_rt_idx)[0]
            if add_cid_by_tt:
                data["tt_id"] = rt_doc_id

            data["InChIKey"] = mol_attr["_id"]
            for k in ["SMILES", "formula", "scaffold", "is_tautomers"]:
                data[k] = mol_attr[k]

            # mol attrs
            data["mol_attrs"]["rt"] = self._rt_data.loc[rt_doc_id, "rt"]

            for k in ["wt", "volume", "n_hba", "n_hbd", "n_ring",
                      "n_aromatic_ring", "n_aliphatic_ring", "mLogP"]:
                data["mol_attrs"][k] = mol_attr[k]

            # node attrs
            atom_attrs = pd.DataFrame.from_records(
                mol_attr["atom_attrs"]['atom_attrs']
            )

            data["node_attrs"]["symbol"] = atom_attrs.symbol.tolist()
            data["node_attrs"]["hybridization"] = atom_attrs\
                .hybridization.tolist()
            data["node_attrs"]["degree"] = atom_attrs.degree.tolist()
            data["node_attrs"]["n_hs"] = atom_attrs.n_hs.tolist()
            data["node_attrs"]["formal_charge"] = atom_attrs\
                .formal_charge.tolist()
            data["node_attrs"]["aromaticity"] = atom_attrs.aromaticity.tolist()
            data["node_attrs"]["in_ring"] = atom_attrs.in_ring.tolist()

            distance = pd.DataFrame.from_records(
                mol_attr["atom_attrs"]['distance']
            )
            if self._knn_to_vec:
                knn_symb, knn_dist = select_by_k_dist(
                    distance,
                    k=self._k,
                    knn_to_vec=self._knn_to_vec,
                    max_distance=self._max_distance
                )
                data["knn_attrs"]["knn"] = knn_symb
                data["knn_attrs"]["distance"] = knn_dist
            else:
                data["knn_attrs"]["knn"] = select_by_k_dist(
                    distance,
                    k=self._k,
                    knn_to_vec=self._knn_to_vec,
                    max_distance=self._max_distance
                )

            # edge attrs
            edge_attrs = pd.DataFrame.from_records(
                mol_attr["edge_attrs"]["edge_attrs"]
            )
            data["edge_attrs"]["bondtype"] = edge_attrs.bondtype.tolist()
            data["edge_attrs"]["conjugation"] = edge_attrs.conjugation.tolist()
            data["edge_attrs"]["in_ring"] = edge_attrs.in_ring.tolist()
            data["edge_attrs"]["stereos"] = edge_attrs.stereos.tolist()
            data["edge_attrs"]["aromatic"] = edge_attrs.aromatic.tolist()

            edge_index = pd.DataFrame.from_records(
                mol_attr["edge_attrs"]["edge_index"]
            )
            edge_index = np.array(
                edge_index.loc[:, ["begin", "end"]]
            ).transpose()
            data["edge_index"] = torch.tensor(edge_index)
            return(data)

        if not(knn_to_vec is None):
            self._knn_to_vec = knn_to_vec

        if not(max_distance is None):
            self._max_distance

        self._k = k

        # Query RT data
        rt_data_ids = self._rt_data.index.tolist()
        if not(n_mol is None):
            rt_data_ids = sample(rt_data_ids, n_mol)

        if query is None:
            if self._rt_query is None:
                return None
            else:
                attr_query = {
                    "rt_data_id": {
                        "$elemMatch": {
                            "$in": rt_data_ids
                        }
                    },
                    "atom_attrs.embedding_flag": 0
                }
                self._attr_query = attr_query
        else:
            attr_query = {
                    "rt_data_id": {
                        "$elemMatch": {
                            "$in": rt_data_ids
                        }
                    },
                    "atom_attrs.embedding_flag": 0
                }
            if len(attr_query.keys() & query.keys()) > 1:
                print("repetitive keys in default query and input query.")

            if version_info.major >= 3 and version_info.minor >= 9:
                self._attr_query = query | attr_query
            else:
                self._attr_query = {**query, **attr_query}

        # Query molecular property
        if limit is None:
            n_total_mol = self._mng_col_attr.count_documents(self._attr_query)
            mol_attr_cursor = self._mng_col_attr\
                .find(self._attr_query)
        else:
            n_total_mol = min([
                self._mng_col_attr.count_documents(self._attr_query),
                limit
            ])
            mol_attr_cursor = self._mng_col_attr\
                .find(self._attr_query)\
                .limit(n_total_mol)

        data_rt_idx = set(rt_data_ids)
        self._data_list = n_total_mol*[None]
        
        for i, mol_attr in tqdm(
                enumerate(mol_attr_cursor),
                desc="Reading Data From Mongod",
                total=n_total_mol):
            self._data_list[i] = gen_data(mol_attr, data_rt_idx, add_cid_by_tt)

        if add_cid_by_tt:
            cntr = collections.Counter(
                [item["tt_id"] for item in self._data_list]
            )
            for i in range(len(self._data_list)):
                self._data_list[i]["n_tt"] = cntr[self._data_list[i]["tt_id"]]


class RTData(InMemoryDataset):
    def __init__(
            self, root: str, profile_name: str = "profile.json",
            conn_str: Union[None, str] = None,
            host: Union[None, str] = "mongodb://localhost",
            port: Union[None, int] = 27017,
            split: str = "train", transform: Callable = None,
            pre_transform: Callable = None, pre_filter: Callable = None,
            splitter: DataSplitter = RandomSplitter):
        """
        splitter: A DataSplitter object that use to split data
                  into different sets
        """
        self.splitter = splitter
        self.profile_name = profile_name
        self.profile_mongo = None
        self.profile_rt = None
        self.profile_mol_attrs = None
        self.mgMlAttrRdr = None
        self._mgConnStr = conn_str
        if conn_str is None:
            self._mgHost = host
            self._mgPort = port
        else:
            self._mgHost = None
            self._mgPort = None

        super().__init__(root, transform, pre_transform, pre_filter)

        if split == "train":
            path = self.processed_paths[0]
        elif split == "valid":
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        else:
            raise ValueError(f"split variables should be \
                either 'train', 'validate', or 'test'")
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return self.profile_name

    @property
    def processed_file_names(self):
        return ["train.pt", "valid.pt", "test.pt"]

    @property
    def mng_connecting_string(self) -> str:
        str_sub('//.+:.+@', "//*****:********@", self._mgConnStr)

    def process(self) -> None:
        step = 1
        print(f"  > {step}. Read profile")
        step += 1
        with open("/".join(
                [self.root, "raw", self.raw_file_names]), "r"
                ) as json_file:
            profile = json.load(fp=json_file)
            self.profile_mongo = profile["mongo"]
            self.profile_rt = profile["rt"]
            self.profile_mol_attrs = profile["attrs"]

        conn_str = str_sub('//.+:.+@', "//*****:********@", self._mgConnStr)
        print(f"  > {step}. Connect to {conn_str}")
        step += 1
        self.mgMlAttrRdr = MongodReader(
            conn_str=self._mgConnStr,
            host=self._mgHost,
            port=self._mgPort,
            db=self.profile_mongo["db"],
            col_rt=self.profile_mongo["collection_rt_data"],
            col_attr=self.profile_mongo["collection_attrs"]
        )
        self._mgConnStr = conn_str

        print(f"  > {step}. Read data")
        step += 1

        profile_rt = self.profile_rt
        self.mgMlAttrRdr.read_rt(
            query=profile_rt["query"],
            exclude_cols=profile_rt["exclude_cols"]
        )

        profile_attrs = self.profile_mol_attrs
        self.mgMlAttrRdr.read_mol_attrs(
            query=profile_attrs["query"],
            limit=profile_attrs["limit"],
            n_mol=profile_attrs["n_mol"],
            add_cid_by_tt=profile_attrs["mol"]["add_cid_by_tt"],
            knn_to_vec=profile_attrs["node"]["knn_to_vec"],
            k=profile_attrs["node"]["k"],
            max_distance=profile_attrs["node"]["max_distance"]
        )

        data_list = self.mgMlAttrRdr.data_list

        if self.pre_filter is not None:
            print(f"  > {step}. Data pre-filter")
            step += 1
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            print(f"  > {step}. Data pre-transform")
            step += 1
            data_list = [self.pre_transform(data) for data in data_list]

        print(f"  > {step}. Split data (num_data: {len(data_list)})")
        step += 1
        train_set, valid_set, test_set = self.splitter(data_list)

        print(f"  > {step}. Save data")
        print(f"  > {step}.1 Training set ({len(train_set)})")
        train_data, train_slices = self.collate(train_set)
        torch.save((train_data, train_slices), self.processed_paths[0])
        print(f"  > {step}.2 Validation set ({len(valid_set)})")
        valid_data, valid_slices = self.collate(valid_set)
        torch.save((valid_data, valid_slices), self.processed_paths[1])
        print(f"  > {step}.3 Test set ({len(test_set)})")
        test_data, test_slices = self.collate(test_set)
        torch.save((test_data, test_slices), self.processed_paths[2])
        step += 1
        return None


# %%
if __name__ == '__main__':
    mgMlAttrRdr = MongodReader(
        host="mongodb://127.0.0.1",
        port=27017,
        db="mols",
        col_rt="RT_data",
        col_attr="MolAttrs"
    )

    mgMlAttrRdr.read_rt(
        query={
            "database": "SMRT",
            "dataset": "SMRT"
        },
        exclude_cols=[
            "SMILE", "InChI", "unit", "dataset", "database",
            "username", "upload_date", "name", "PubChemCid"
        ]
    )

    mgMlAttrRdr.read_mol_attrs(
        query=None,
        n_mol=2000,
        limit=1000,
        k=5,
        knn_to_vec=True,
        max_distance=None,
        add_cid_by_tt=True
    )
