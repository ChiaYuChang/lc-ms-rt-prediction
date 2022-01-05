import json
import pandas as pd
import pickle
import numpy as np
import zlib
import torch

from .ParSet import LinearLayerPars

from copy import deepcopy
from datetime import datetime
from hashlib import sha256
from random import choices
from string import ascii_letters, digits
from torch_geometric.data.data import Data
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.utils import to_dense_adj
from typing import Union, List, Dict, OrderedDict


def iqr(x: Union[pd.Series, np.array]):
    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    """Vectorized interquartile range (IQR)"""
    return x.quantile(.75) - x.quantile(.25)


def predictor_par_transform(
        in_features: int = 64,
        hidden_features: Union[List[int], None] = None,
        out_features: int = 1,
        dropout: Union[List[float], float] = 0.1,
        relu: Union[List[bool], bool] = False,
        batch_norm: Union[List[bool], bool] = False
        ) -> OrderedDict:

    in_features = [in_features] + hidden_features
    out_features = hidden_features + [out_features]
    n_layer = len(in_features)

    # cast dropout, relu, and batch_norm
    if isinstance(dropout, float):
        dropout = [dropout] * n_layer
        dropout[-1] = -1
    if isinstance(relu, bool):
        relu = [relu] * n_layer
        relu[-1] = False
    if isinstance(batch_norm, bool):
        batch_norm = [batch_norm] * n_layer
        batch_norm[-1] = False

    prdctr_lyr_pars = OrderedDict()
    for i in range(n_layer):
        prdctr_lyr_pars[i] = LinearLayerPars(
            in_features=in_features[i],
            out_features=out_features[i],
            dropout=(dropout[i], True),
            relu=relu[i],
            batch_norm=batch_norm[i]
        )
    return prdctr_lyr_pars


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

    data_body = Data()
    
    data_sup = {}
    for k in data.keys:
        if k == sup_data_key:
            data_sup = add_sup_field | data[k]
            data_sup["rt"] = float(data.y)
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


def doc_to_data(doc: Dict, w_sup_field: bool = False, suppress_error: bool = False) -> Data:
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
    return data


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
        which: str = "edge_index", to_sparse: bool = True
        ) -> torch.Tensor:
    A = torch.squeeze(to_dense_adj(data[which])).float()
    if self_loop:
        A = A + torch.eye(A.shape[1])
        A = (A > 0).float()
    D = torch.diag(torch.sqrt(1/torch.sum(A, 1, dtype=torch.float)))
    
    if to_sparse:
        return torch.linalg.multi_dot([D, A, D]).to_sparse()
    else:
        return torch.linalg.multi_dot([D, A, D])
