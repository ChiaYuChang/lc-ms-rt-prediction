import torch
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Union, Callable, NamedTuple
from torch_geometric.data import Data, InMemoryDataset
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

class DataTransformer():

    def __init__(
            self, transform_list: Union[List[Transform], Transform],
            inplace: bool = False) -> None:
        if isinstance(transform_list, Transform):
            transform_list = [transform_list]
        self.transform_list = transform_list
        self.inplace = inplace
    
    def __call__(self, data: Union[List[Data], Data, InMemoryDataset]) -> Union[List[Data], Data]:
        if isinstance(data, Data):
            return self.process(data=data)
        elif isinstance(data, Iterable):
            return [self.process(d) for d in data]
        else:
            raise TypeError

    def process(self, data) -> Data: 
        if not(self.inplace):
            data = deepcopy(data)

        data_fields = set(data.keys)
        for t in self.transform_list:
            if t.name in data_fields:
                data[t.name] = self.merge(
                    existed_value=data[t.name],
                    new_value=t.func(data, **t.args))
            else:
                data[t.name] = t.func(data, **t.args)
        return data
    
    def merge(self, existed_value, new_value):
        return torch.cat((existed_value, new_value), axis=1)

    def __repr__(self) -> str:
        return f"DataTransformer ({len(self.transform_list)})"

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