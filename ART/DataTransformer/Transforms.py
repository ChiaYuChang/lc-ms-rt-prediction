import torch
import numpy as np

from typing import Any, Dict, Callable, Optional
from scipy.stats import norm
from ART.Data import GraphData as Data
from ART.funcs import calc_knn_graph, calc_radius_graph, calc_dist_bw_node, calc_tilde_A
from ART.funcs import hop

class Transform():

    def __init__(self, name:str, func: Callable[[Data], Any], args: Dict = {}):
        self._name = name
        self._func = func
        self._args = args
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        self._name = name

    @property
    def func(self):
        return self._func
    
    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, args):
        self._args = {**self._args, **args}

    def __repr__(self) -> str:
        out = f"Transform{{name={self.name}" +\
            f", func={self.func.__name__}" +\
            f", args={self.args.__repr__()}}}"
        return out


def gen_discance_attr(data: Data, which: str="edge_index"):
    return calc_dist_bw_node(
        data=data, edge_index=data[which])


gen_bond_length = Transform(
    name="edge_attr",
    func=gen_discance_attr,
    args={"which": "edge_index"}
)

gen_knn_graph = Transform(
    name="knn_edge_index",
    func=calc_knn_graph,
    args={"k": 3}
)

gen_knn_distance = Transform(
    name="knn_edge_attr",
    func=gen_discance_attr,
    args={"which": gen_knn_graph.name}
)

gen_radius_graph = Transform(
    name="radius_edge_index",
    func=calc_radius_graph,
    args={"radius": 1.5}
)

gen_radius_distance = Transform(
    name="radius_edge_attr",
    func=gen_discance_attr,
    args={"which": gen_radius_graph.name}
)

gen_normalized_adj_matrix = Transform(
    name="normalized_adj_matrix",
    func=calc_tilde_A,
    args={"self_loop": True, "which": "edge_index", "to_coo": True}
)

def calc_mw_ppm(data, mw_list, scaler: Optional[float] = 1e6, use_torch: bool = True):
    if use_torch:
        return torch.abs(mw_list - data.sup["mw"])/ mw_list * scaler
    else:
        if isinstance(mw_list, torch.Tensor):
            mw_list = mw_list.numpy()
        return np.abs(mw_list - data.sup["mw"])/ mw_list * scaler

# def calc_mw_mask(
#         data, thr: Optional[float] = 20.0,
#         mask_dtype: Optional[torch.dtype] = torch.float32,
#         ppm_attr_name: Optional[str] = "mw_ppm"):
#     return (data[ppm_attr_name] <= thr).type(mask_dtype)

def calc_mw_mask(
        data, thr: Optional[float] = 20.0,
        mask_dtype: Optional[torch.dtype] = torch.float32,
        shift: Optional[float] = 0,
        ppm_attr_name: Optional[str] = "mw_ppm",
        use_torch: bool = True):
    ppm = data[ppm_attr_name]
    if isinstance(ppm, torch.Tensor):
        ppm = ppm.numpy()
    mask = norm.pdf(ppm, loc=0, scale=thr)
    mask = mask/np.max(mask) + shift
    if use_torch:
        return torch.tensor(mask, dtype=mask_dtype)
    else:
        return mask

gen_mw_ppm = Transform(
    name="mw_ppm",
    func=calc_mw_ppm,
    args={"mw_list": None, "scaler": 1e6}
)

gen_mw_mask = Transform(
    name="mw_mask",
    func=calc_mw_mask,
    args={"thr": 20, "mask_dtype": torch.float32, "ppm_attr_name": "mw_ppm", "use_torch": True}
)

def n_hop(data, n: int = 1, rm_self_loop: bool = True):
    n_hop_edge_index = data.edge_index
    for i in range(n): 
        n_hop_edge_index = hop(n_hop_edge_index, data.num_nodes, rm_self_loop)
    return n_hop_edge_index

gen_n_hop_edge_index = Transform(
    name="n_hop_edge_index",
    func=n_hop,
    args={"n": 1, "rm_self_loop": True}
)