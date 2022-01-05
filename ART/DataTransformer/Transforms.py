from typing import Any, Dict, Callable
from torch_geometric.data import Data
from ART.funcs import calc_knn_graph, calc_radius_graph, calc_dist_bw_node, calc_tilde_A


class Transform():

    def __init__(self, name:str, func: Callable[[Data], Any], args: Dict):
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
    args={"r": 1.5}
)

gen_radius_distance = Transform(
    name="radius_edge_attr",
    func=gen_discance_attr,
    args={"which": gen_radius_graph.name}
)

gen_normalize_adj_matrix = Transform(
    name="normalize_adj_matrix",
    func=calc_tilde_A,
    args={"self_loop": True, "which": "edge_index"}
)
