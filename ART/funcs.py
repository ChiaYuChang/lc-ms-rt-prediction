import json
import os
import pandas as pd
import pickle
import numpy as np
import torch
import zlib

from copy import deepcopy
from datetime import datetime
from hashlib import sha256
from numpy.lib.arraysetops import isin
from random import choices
from string import ascii_letters, digits
# from torch_geometric.data.data import Data
from ART.Data import GraphData as Data
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.utils import to_dense_adj
from typing import Union, List, Dict, OrderedDict, Optional


def iqr(x: Union[pd.Series, np.array]):
    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    """Vectorized interquartile range (IQR)"""
    return x.quantile(.75) - x.quantile(.25)


def json_snapshot_to_doc(
        json_snapshot: dict,
        compress: bool = True
        ) -> dict:
    str_snapshot = json.dumps(json_snapshot)
    binary_snapshot = str_snapshot.encode()
    binary_sha256 = sha256(binary_snapshot)

    if compress is True:
        snapshot = zlib.compress(binary_snapshot)
        compressed_by = "zlib"
    else:
        snapshot = str_snapshot
        compressed_by = None

    doc = {
        "name": json_snapshot["experiment"]["name"],
        "time": datetime.utcnow(),
        "num_trail": len(json_snapshot["experiment"]["trials"]),
        "sha256": binary_sha256.hexdigest(),
        "compress": (compress, compressed_by),
        "snapshot": snapshot
    }
    return doc


def doc_to_json_snapshot(doc: dict) -> dict:
    snapshot = doc["snapshot"]
    snapshot_time = doc["time"]

    print(
        snapshot_time.strftime(">  Use snapshot - %Y-%m-%d %H:%M:%S")
    )

    if doc["compress"][0]:
        if doc["compress"][1] == "zlib":
            binary_snapshot = zlib.decompress(snapshot)
            binary_sha256 = sha256(binary_snapshot)

            print(">  Validate SHA256")
            if binary_sha256.hexdigest() != doc["sha256"]:
                return None
    return json.loads(binary_snapshot.decode())


def array_to_tensor(data: Data, zero_thr: float = 0.5, inplace: bool = False) -> Data:
    if not(inplace):
        data = deepcopy(data)    
    
    for key in data.keys:
        if isinstance(data[key], np.ndarray):
            tnsr = torch.from_numpy(data[key])
            if torch.sum(tnsr == 0) / tnsr.numel() > zero_thr:
                tnsr = tnsr.to_sparse()
            data[key] = tnsr
    return data


def tensor_to_array(data: Data, inplace: bool = False) -> Data:
    if not(inplace):
        data = deepcopy(data)    
    
    for key in data.keys:
        if isinstance(data[key], torch.Tensor):
            if data[key].is_sparse:
                data[key] = data[key].to_dense().numpy()
            else:
                data[key] = data[key].numpy()
    return data


def split_list(x: List, n: int, return_idxs: bool = False):
    x_len = len(x)
    chunk_size = x_len//n
    remainder = x_len % n
    start = [0] + [None] * (n-1)
    for i in range(1, n):
        start[i] = start[i-1] + chunk_size
        if i <= remainder:
            start[i] = start[i] + 1
    end = start[1:] + [x_len]

    if return_idxs:
        return [(s, e) for s, e in zip(start, end)]
    else:
        return [x[s:e] for s, e in zip(start, end)]


def data_to_doc(
        data: Data,
        compress: bool = True,
        sup_data_key: str = "sup",
        add_sup_field: Dict = {}
        ) -> dict:
    data = tensor_to_array(data=data, inplace=False)
    data_body = Data()
    data_sup = {}
    for k in data.keys:
        if k == sup_data_key:
            data_sup = {**add_sup_field, **data[k]}
        else:
            data_body[k] = data[k]

    binary_data_body = pickle.dumps(data_body)
    binary_data_body_sha256 = sha256(binary_data_body)
    if compress is True:
        binary_data_body = zlib.compress(binary_data_body)
        compressed_by = "zlib"
    else:
        compressed_by = None
    doc = data_sup
    doc["binary_data"] = binary_data_body
    doc["sha256"] = binary_data_body_sha256.hexdigest()
    doc["compress"] = (compress, compressed_by)
    doc["time"] = datetime.utcnow()
    doc["sup_data_key"] = sup_data_key
    doc["upload_date"] = datetime.strptime(
        doc["upload_date"] + "T00:00:00+00:00", "%Y-%m-%dT%H:%M:%S%z"
    )
    return doc


def doc_to_data(
        doc: Dict, w_sup_field: bool = False, 
        suppress_error: bool = False, **kwarg) -> Data:
    binary_data = doc["binary_data"]
    if doc["compress"][0]:
        if doc["compress"][1] == "zlib":
            binary_data = zlib.decompress(binary_data)
            binary_data_sha256 = sha256(binary_data)
            if binary_data_sha256.hexdigest() != doc["sha256"]:
                if suppress_error:
                    return None
                else:
                    print("sha256 not match")
                    raise ValueError
    data = pickle.loads(binary_data)

    if w_sup_field:
        sup_field = {}
        exclude_set = set(["binary_data", "sha256", "compress", "time", "sup_data_key", "rt"])
        for k in doc.keys():
            if k not in exclude_set:
                sup_field[k] = doc[k]
        data[doc["sup_data_key"]] = sup_field
    return array_to_tensor(data = data, inplace=True, **kwarg)


def gen_random_str(n):
    return ''.join(choices(ascii_letters + digits, k=n))


def calc_dist_bw_node(data: Data, edge_index) -> torch.Tensor:
    return torch.norm(
        data.pos[edge_index[0, :]] - data.pos[edge_index[1, :]],
        p=2, dim=-1
    ).view(-1, 1)


def calc_radius_graph(
        data: Data, radius: float) -> torch.Tensor:
    if "pos" in data.keys:
        return radius_graph(data.pos, r=radius)
    else:
        print("pos should be included in Data")
        raise ValueError


def calc_knn_graph(data: Data, k: int) -> torch.Tensor:
    if "pos" in data.keys:
        return knn_graph(data.pos, k=k)
    else:
        print("pos should be included in Data")
        raise ValueError


def calc_tilde_A(
        data, self_loop: bool= True,
        which: str = "edge_index",
        to_sparse: bool = True,
        to_coo: bool = True
        ) -> torch.Tensor:
    A = torch.squeeze(to_dense_adj(data[which])).float()
    if self_loop:
        A = A + torch.eye(A.shape[1])
        A = (A > 0).float()
    D = torch.diag(torch.sqrt(1/torch.sum(A, 1, dtype=torch.float)))
    
    tilda_A = torch.linalg.multi_dot([D, A, D])
    if to_sparse:
        if to_coo:
            tilda_A = tilda_A.to_sparse()
            return {"index": tilda_A.indices(), "value": tilda_A.values()}
        else:
            return tilda_A.to_sparse()
    else:
        if to_coo:
            print("to_coo only works when to_sparse is true")
        return tilda_A


def np_one_hot(x: np.ndarray, num_classes:int = -1):
    if num_classes < 0:
        num_classes = np.max(x)
    
    if len(x.shape) == 0:
        y = np.zeros((num_classes))
        y[int(x-1)] = 1.0
        return y
    elif len(x.shape) == 1:
        y = np.zeros((x.shape[0], num_classes))
        for i in range(len(x)):
            print(f"({i}, {x[i]-1})")
            y[i, int(x[i]-1)] = 1.0
        return y
    else:
        print(f"The one_hot function only works for scalar and 1D array but x is a {len(x.shape)}D array.")
        raise ValueError


def str_padding(x: str, length: int, character: str = ".", after: bool = True):
    if len(x) > length:
        return x
    
    if len(character) == 1:
        if after:
            return x + " " + character * (length - len(x)-1)
        else:
            return character * (length - len(x)-1) + " " + x
    else:
        m = (length - len(x) - 1) // len(character)
        r = (length - len(x) - 1) % len(character)
        p = character * m + character[0:r]
        if after:
            return x + " " + p
        else:
            return p + " " + x


def check_has_processed(
        root: str, raw_file_names: Union[List[str], str],
        processed_file_names: Union[List[str], str] = [
            "pre_filter.pt",  "pre_transform.pt",  "test.pt",  "train.pt",  "valid.pt"],
        indent: str="\t"
    ):
    prefix = "    - "
    p_len = 50
    
    print(indent + "1. Checking root ...")
    if os.path.isdir(root):
        print(indent + prefix + str_padding("root directory", length=p_len) + " found")
    else:
        raise IOError("Root directory does not exist.")

    if isinstance(raw_file_names, str):
        raw_file_names = [raw_file_names]
    
    print(indent + "2. Checking raw files ...")
    if os.path.isdir(root + "/raw"):
        print(indent + prefix + str_padding("raw file directory", length=p_len) + " found")
        for raw_file_name in raw_file_names:
            if not(os.path.isfile(root + "/raw/" + raw_file_name)):
                print(indent + prefix + str_padding(raw_file_name, length=p_len) + " not found")
                raise IOError
            else:
                print(indent + prefix + str_padding(raw_file_name, length=p_len) + " found")
    else:
        print(indent + prefix + str_padding("raw file directory", length=p_len) + " not found")
        raise IOError

    if isinstance(processed_file_names, str):
        processed_file_names = [processed_file_names]

    print(indent + "3. Checking processed files ...")
    flag = True
    if os.path.isdir(root + "/processed"):
        print(indent + prefix + str_padding("processed file directory", length=p_len) + " found")
        for processed_file_name in processed_file_names:
            if not(os.path.isfile(root + "/processed/" + processed_file_name)):
                print(indent + prefix + str_padding(processed_file_name, length=p_len) + " not found")  
                flag = False
            else:
                print(indent + prefix + str_padding(processed_file_name, length=p_len) + " found")  
    else:
        print(indent + prefix + str_padding("processed file directory", length=p_len) + " not found")
        flag = False
    
    return flag


def hop(edge_index: torch.Tensor,
        num_nodes: Optional[int] = None,
        rm_self_loop: Optional[bool] = True
        ) -> torch.Tensor:
    device = edge_index.device   
    if num_nodes is None:
        num_nodes = edge_index.max() + 1
    m = torch.sparse_coo_tensor(
        indices=edge_index,
        values=torch.ones(edge_index.shape[1]).to(device),
        size=(num_nodes, num_nodes)
    )
    m_hop = (m + torch.sparse.mm(m, m)).coalesce().indices()
    if rm_self_loop:
        m_hop = m_hop[:, m_hop[0, :] != m_hop[1, :]]
    return m_hop