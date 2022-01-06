import pandas as pd
import torch
import itertools
import numpy as np

from .funcs import iqr
from collections import defaultdict
from copy import deepcopy
from torch import Tensor
from torch.nn.functional import one_hot
from torch_geometric.data import Data, Dataset
from torch_geometric.data.dataset import Dataset
from typing import Callable, List, Union
from tqdm import tqdm

class DataSummarizer:
    def __init__(
            self,
            data_list: Union[List[Data], Dataset],
            attrs_group: str,
            discrete_vars: List[str] = [],
            discrete_thr: Union[List[float], float] = 0.05,
            continuous_vars: List[str] = [],
            continuous_funs: List[Callable] = [
                np.mean, np.median, max, min, np.std, iqr
            ],
            rm_tt: bool = True,
            id_var: str = "SMILES",
            verbose: bool = False,
            other_str: str = "Other"
            ) -> None:
        self._dscrt_vars = discrete_vars
        self._dscrt_vars_smry = {}
        self._dscrt_vars_map_func = {}
        if len(self._dscrt_vars) > 0:
            for var in discrete_vars:
                self._dscrt_vars_smry[var] = {}
                self._dscrt_vars_map_func[var] = {}
        self._level_threshold = discrete_thr
        self._attrs_group = attrs_group
        self._cntns_vars = continuous_vars
        self._cntns_vars_smry = {}
        self._cntns_vars_map_func = {}
        if len(self._cntns_vars) > 0:
            for var in continuous_vars:
                self._cntns_vars_smry[var] = {}
                self._cntns_vars_map_func[var] = {}
        self._summary_funs = continuous_funs
        self._rm_tt = rm_tt
        self._id_var = id_var
        self._verbose = verbose
        self._other_str = other_str
        self.__process(data=self.data_extract(data_list=data_list))

    @property
    def vars(self) -> dict:
        return {"discrete": self._dscrt_vars, "continuous": self._cntns_vars}

    @property
    def id_var(self) -> str:
        return self._id_var

    @property
    def attrs_group(self) -> str:
        return self._attrs_group

    @property
    def discrete_vars(self) -> Union[List[str], None]:
        return self._dscrt_vars

    @property
    def discrete_vars_smry(self):
        return self._dscrt_vars_smry

    @property
    def continuous_vars(self) -> Union[List[str], None]:
        return self._cntns_vars

    @property
    def continuous_vars_smry(self):
        return self._cntns_vars_smry

    def show_mappers(self) -> None:
        for var in self._dscrt_vars:
            entropy = self._dscrt_vars_smry[var]["entropy"]
            star = "*" if entropy < 0.5 else ""
            print(f"{star}{var} (Entropy: {entropy:05.2f})")
            for lev in self._dscrt_vars_smry[var]["levels"]:
                freq = self._dscrt_vars_smry[var]["count"].\
                    loc[lev, "relative_freq"]
                print(f"  - {str(lev):10s} ({freq:05.2f}%) -> {self.map(var, [lev])}")
            print("\n")
        return None

    def show_normalizer(self) -> None:
        for key in self._cntns_vars_smry.keys():
            mu = round(self._cntns_vars_smry[key]["mean"], 4)
            theta = round(self._cntns_vars_smry[key]["std"], 4)
            print(key)
            print(f"  - mu: {mu:8.4f}, theta: {theta:8.4f}")
            print("\n")

    def data_extract(
            self, data_list: Union[List[Data], Dataset]
            ) -> pd.DataFrame:

        step_cntr = 1
        if self._rm_tt:
            if self._verbose is True:
                print(f"{step_cntr}. Filter data")
                step_cntr += 1
            data_list = [
                data for data in tqdm(data_list, total=len(data_list))
                if bool(data["is_tautomers"]) is False
            ]

        if self._verbose is True:
            print(f"{step_cntr}. Set up empty dictionary")
            step_cntr += 1
        all_vars = self.discrete_vars + self.continuous_vars
        all_vars_dict = {}
        for var in all_vars:
            all_vars_dict[var] = [None] * len(data_list)
        all_vars_dict[self.id_var] = [None] * len(data_list)

        if self._verbose is True:
            print(f"{step_cntr}. Extract data ({len(data_list)})")
            step_cntr += 1
        for idx, item in enumerate(tqdm(data_list, total=len(data_list))):
            all_vars_dict[self.id_var][idx] = item[self.id_var]
            for var in all_vars:
                all_vars_dict[var][idx] = item[self.attrs_group][var]

        first_var = all_vars[0]
        if isinstance(all_vars_dict[first_var][0], List):
            if self._verbose is True:
                print(f"{step_cntr}. Concatenate data")
                step_cntr += 1
            all_vars_dict[self.id_var] = np.repeat(
                all_vars_dict[self.id_var],
                repeats=[len(item) for item in all_vars_dict[all_vars[0]]],
                axis=0)
            for var in all_vars:
                all_vars_dict[var] = list(
                    itertools.chain.from_iterable(all_vars_dict[var])
                )

        if (isinstance(all_vars_dict[first_var][0], Tensor) and
                (all_vars_dict[first_var][0].dim() == 1)):
            for var in all_vars:
                all_vars_dict[var] = torch.concat(all_vars_dict[var]).tolist()
        return pd.DataFrame(all_vars_dict)

    def __process(self, data: pd.DataFrame) -> None:
        if len(self.continuous_vars) > 0:
            self.__summarzie_continuous_vars(data)
        if len(self.discrete_vars) > 0:
            self.__summarize_discrete_vars(data)
        return None

    def map(self, var: str, keys: Union[List, str, int, float, np.ndarray]) -> Tensor:
        if not(isinstance(keys, List)):
            keys = [keys]

        if var in self._dscrt_vars:
            return self._dscrt_vars_map_func[var](keys)
        elif var in self._cntns_vars:
            return self._cntns_vars_map_func[var](keys)
        else:
            raise ValueError("Variable not found.")

    def __call__(
            self, data: Union[dict, Data],
            vars: Union[List[str], None] = None,
            to_one_hot: bool = False, concat: bool = False,
            to_Data: bool = False
            ) -> Union[dict, Data, Tensor]:

        data = deepcopy(data[self.attrs_group])

        if vars is None:
            vars = self._dscrt_vars + self._cntns_vars

        new_data = {}

        for var in vars:
            tmp = self.map(var, data[var])
            if (var in self._dscrt_vars) and (to_one_hot is True):
                levels = self.discrete_vars_smry[var]["levels"]
                num_class = len(levels)
                if not(self._other_str in levels) and (num_class > 2):
                    num_class += 1
                tmp = one_hot(tmp, num_class)
            new_data[var] = tmp.float()

        if concat is True:
            d_vars = list(set(vars) & set(self.discrete_vars))
            vals = [None] * len(d_vars)
            for i, v in enumerate(d_vars):
                vals[i] = new_data.pop(v)
            vals_dim = vals[0].dim()
            new_data["concat"] = torch.cat(tensors=vals, dim=vals_dim-1)

        if to_Data is True:
            new_data = Data.from_dict(new_data)

        return new_data

    def __summarize_discrete_vars(self, data: pd.DataFrame) -> None:
        """Summarizing discrete molecular attributes."""

        threshold = self._level_threshold
        if isinstance(threshold, float):
            threshold = [threshold] * len(self.discrete_vars)
            self._level_threshold = threshold

        for thr in threshold:
            if thr >= 1 or thr < 0:
                raise ValueError("Threshold shold be within [0, 1)")

        for d_var, thr in zip(self._dscrt_vars, threshold):
            self._dscrt_vars_smry[d_var] = {}
            self._dscrt_vars_smry[d_var]["mode"] = data[d_var].mode().iloc[0]
            self._dscrt_vars_smry[d_var]["ori_levels"] = data[d_var]\
                .unique().tolist()
            cnt = data.groupby([d_var])[d_var].agg([("count", len)])
            cnt.index.names = ["n"]
            cnt["relative_freq"] = np.round(
                cnt["count"] / cnt["count"].sum() * 100, 2
            )
            cnt["label"] = cnt.index
            thr = max(cnt["relative_freq"]) * thr

            cnt.loc[cnt[cnt.relative_freq < thr].index, "label"] = self._other_str           
            cnt = cnt.groupby("label")[["count", "relative_freq"]].agg(sum)

            cnt.index.names = ["n"]
            self._dscrt_vars_smry[d_var]["entropy"] = - sum(
                np.log2(cnt.relative_freq/100) * cnt.relative_freq/100
            )
            if self._other_str not in set(cnt.index):
                if not(set(cnt.index) == set([True, False])):
                    new_row = pd.DataFrame(
                        {"label": ["Other"], "count": [0], "relative_freq": [0.0]
                    })
                    new_row.set_index(keys="label", inplace=True)
                    cnt = cnt.append(new_row)
            self._dscrt_vars_smry[d_var]["levels"] = cnt.index.tolist()
            self._dscrt_vars_smry[d_var]["threshold"] = thr
            self._dscrt_vars_smry[d_var]["count"] = cnt
            
            self._dscrt_vars_map_func[d_var] = self.__mapper_factory(cnt)

        return self._dscrt_vars_smry

    def __summarzie_continuous_vars(self, data: pd.DataFrame) -> None:
        """Summarizing continuous molecular attributes."""
        funs = self._summary_funs
        vars_smry = data.loc[:, self.continuous_vars].agg(funs)
        self._cntns_vars_smry = vars_smry.to_dict()

        for key in self._cntns_vars_smry.keys():
            mu = self._cntns_vars_smry[key]["mean"]
            theta = self._cntns_vars_smry[key]["std"]
            self._cntns_vars_map_func[key] = self.__normalizer_factory(mu, theta)

        return self._cntns_vars_smry

    def __mapper_factory(
            self, attr_count: pd.DataFrame
            ) -> Callable[[List], Tensor]:
        attr_dict = defaultdict(lambda: 0)

        indexs = attr_count.index.to_list()

        if set([False, True]) == set(indexs):
            n_level = 0
            indexs.sort()
        else:
            if self._other_str not in set(indexs):
                indexs.append(self._other_str)
            n_level = 1

        for idx in indexs:
            if idx != self._other_str:
                attr_dict[idx] = n_level
                n_level += 1

        def mapper(keys: List) -> Tensor:
            values = [None] * len(keys)
            for i, k in enumerate(keys):
                values[i] = torch.tensor(attr_dict[k])
            if len(keys) == 1:
                values = values[0]
            else:
                values = torch.stack(values)
            return values

        return mapper

    def __normalizer_factory(
            self, mu, theta, min=None, max=None
            ) -> Callable[[List], Tensor]:
        # def normalizer(keys: List) -> Tensor:
        #     values = [None] * len(keys)
        #     for i, k in enumerate(keys):
        #         values[i] = (k - mu)/theta
        #     if len(keys) == 1:
        #         values = values[0]
        #     return torch.tensor(values)
        def normalizer(keys: Union[List, np.ndarray]) -> Tensor:            
            if all([isinstance(e, Tensor) for e in keys]):
                keys_shape = [len(keys)] + list(keys[0].shape)
                keys = torch.concat(keys).reshape(keys_shape)

            if isinstance(keys, List):
                keys = torch.tensor(keys)

            values = (keys - mu)/theta
            values = torch.squeeze(values)
            # if values.dim() == 0:
            #     values = torch.unsqueeze(torch.squeeze(values), dim=0)
            return values
        return normalizer


class MolAttrsSummarizer(DataSummarizer):
    def __init__(
            self,
            data_list: Union[List[Data], Dataset],
            attrs_group: str = "mol_attrs",
            discrete_vars: List[str] = [
                "n_hba", "n_hbd", "n_ring",
                "n_aromatic_ring", "n_aliphatic_ring"
            ],
            discrete_thr: Union[List[float], float] = 0.05,
            continuous_vars: List[str] = ["rt", "wt", "volume", "mLogP"],
            continuous_funs: List[Callable] = [
                np.mean, np.median, max, min, np.std, iqr
            ],
            rm_tt: bool = False,
            id_var: str = "SMILES",
            verbose: bool = True,
            other_str: str = "Other"
            ) -> None:

        super().__init__(
            data_list=data_list, id_var=id_var, attrs_group=attrs_group,
            discrete_vars=discrete_vars, discrete_thr=discrete_thr,
            continuous_vars=continuous_vars, continuous_funs=continuous_funs,
            rm_tt=rm_tt, verbose=verbose, other_str=other_str
        )


# %%
class EdgeAttrsSummarizer(DataSummarizer):
    def __init__(
            self,
            data_list: Union[List[Data], Dataset],
            attrs_group: str = "edge_attrs",
            discrete_vars: Union[List[str]] = [
                'bondtype', "conjugation", "in_ring", "stereos", "aromatic"
            ],
            discrete_thr: Union[List[float], float] = 0.05,
            id_var: str = "SMILES",
            rm_tt: bool = True,
            verbose: bool = True,
            other_str: str = "Other"
            ) -> None:
        super().__init__(
            data_list=data_list, id_var=id_var, attrs_group=attrs_group,
            discrete_vars=discrete_vars, discrete_thr=discrete_thr,
            rm_tt=rm_tt, verbose=verbose, other_str=other_str
        )


# %%
class NodeAttrsSummarizer(DataSummarizer):
    def __init__(
            self,
            data_list: Union[List[Data], Dataset],
            attrs_group: str = "node_attrs",
            discrete_vars: Union[List[str]] = [
                "aromaticity", "degree", "formal_charge", "hybridization",
                "in_ring", "n_hs", "symbol"
            ],
            discrete_thr: Union[List[float], float] = 0.05,
            id_var: str = "SMILES",
            rm_tt: bool = True,
            verbose: bool = True,
            other_str: str = "Other"
            ) -> None:

        super().__init__(
            data_list=data_list,
            attrs_group=attrs_group,
            discrete_vars=discrete_vars,
            discrete_thr=discrete_thr,
            rm_tt=rm_tt,
            id_var=id_var,
            verbose=verbose,
            other_str=other_str
        )


# %%
class NodeKnnSummarizer(DataSummarizer):
    def __init__(
            self,
            data_list: Union[List[Data], Dataset],
            attrs_group: str = "knn_attrs",
            symbols_var: str = "knn",
            distance_var: str = "distance",
            token_var: str = "token",
            id_var: str = "SMILES",
            n_bin: int = 10,
            discrete_thr: Union[List[float], float] = 0.05,
            continuous_funs: List[Callable] = [
                np.mean, np.median, max, min, np.std, iqr
            ],
            rm_tt: bool = True,
            verbose: bool = False,
            other_str:str = "Other") -> None:

        self._symbols_var = symbols_var
        self._distance_var = distance_var
        self._token_var = token_var
        self._breaks = []
        self._n_bin = n_bin
        self.k = None
        self.knn_attr = None
        

        discrete_vars = [symbols_var, token_var]
        continuous_vars = [distance_var]

        super().__init__(
            data_list=data_list, attrs_group=attrs_group, id_var=id_var,
            discrete_vars=discrete_vars, discrete_thr=discrete_thr,
            continuous_vars=continuous_vars, continuous_funs=continuous_funs,
            rm_tt=rm_tt, verbose=verbose, other_str=other_str
        )

    @property
    def n_bin(self):
        return self._n_bin

    @property
    def breaks(self):
        return self._breaks

    def show_mappers(self) -> None:
        for var in self._dscrt_vars:
            entropy = self._dscrt_vars_smry[var]["entropy"]
            star = "*" if entropy < 0.5 else ""
            print(f"{star}{var} (Entropy: {entropy:05.2f})")
            for lev in self._dscrt_vars_smry[var]["levels"]:
                freq = self._dscrt_vars_smry[var]["count"].\
                    loc[lev, "relative_freq"]
                print(f"  - {str(lev):10s} ({freq:05.2f}%) \
                    -> {self.map(var, [lev])}")
            print("\n")
        return None

    def tokenizer(
            self,
            symbol: Union[List[str], pd.Series, np.ndarray],
            distance: Union[List[float], pd.Series, np.ndarray]
            ) -> torch.Tensor:
        tokens = self.__tokenizer(symbol, distance)
        return torch.stack(
            [self.map(self._token_var, list(vec)) for vec in tokens]
        )

    def __call__(
            self, data: Union[dict, Data],
            to_one_hot: bool = False,
            to_Data: bool = False
            ) -> Union[dict, Data]:

        data = deepcopy(data[self.attrs_group])

        new_data = {}
        num_node = len(data[self._symbols_var])
        tmp = self.tokenizer(
           symbol=data[self._symbols_var],
           distance=data[self._distance_var]
        )

        if to_one_hot is True:
            levels = self.discrete_vars_smry[self._token_var]["levels"]
            num_class = len(levels)
            if not(self._other_str in levels) and (num_class > 2):
                num_class += 1
            tmp = one_hot(tmp, num_class).float()

        new_data[self._token_var] = tmp.reshape((num_node, -1))

        if to_Data is True:
            new_data = Data.from_dict(new_data)

        return new_data

    def binning_func(
            self, vec: Union[List[float], pd.Series, np.ndarray, float],
            as_str: bool = True) -> np.ndarray:
        if isinstance(vec, float):
            vec = np.array([vec])
        if isinstance(vec, List):
            vec = np.array(vec)
        if isinstance(vec, pd.Series):
            vec = vec.to_numpy()

        if as_str is True:
            return np.digitize(vec, self.breaks).astype(str)
        else:
            return np.digitize(vec, self.breaks)

    def __tokenizer(
            self,
            vec_symbol: Union[List[str], pd.Series, np.ndarray],
            vec_distance: Union[List[float], pd.Series, np.ndarray]
            ) -> np.ndarray:
        if isinstance(vec_symbol, List):
            vec_symbol = np.array(vec_symbol)
        if isinstance(vec_symbol, pd.Series):
            vec_symbol = vec_symbol.to_numpy

        if isinstance(vec_distance, List):
            vec_distance = np.array(vec_distance)
        if isinstance(vec_symbol, pd.Series):
            vec_distance = vec_distance.to_numpy

        vec_dist_binned = self.binning_func(vec_distance, as_str=True)
        vec_symb_bin_dist = np.char.add(vec_symbol, vec_dist_binned)
        return vec_symb_bin_dist

    def data_extract(
            self, data_list: Union[List[Data], Dataset]
            ) -> pd.DataFrame:

        step_cntr = 1
        if self._rm_tt:
            if self._verbose is True:
                print(f"{step_cntr}. Filter data")
                step_cntr += 1
            data_list = [data for data in data_list if data["is_tautomers"] is True]

        if self._verbose is True:
            print(f"{step_cntr}. Set up empty dictionary")
            step_cntr += 1
        all_vars = [self._symbols_var, self._distance_var]
        all_vars_dict = {}
        for var in all_vars:
            all_vars_dict[var] = [None] * len(data_list)
        all_vars_dict[self.id_var] = [None] * len(data_list)

        if self._verbose is True:
            print(f"{step_cntr}. Extract data")
            step_cntr += 1
        for idx, item in enumerate(data_list):
            all_vars_dict[self.id_var][idx] = item[self.id_var]
            for var in all_vars:
                all_vars_dict[var][idx] = item[self.attrs_group][var]

        self.k = len(all_vars_dict[all_vars[0]][0][0])
        all_vars_dict[self.id_var] = np.repeat(
            all_vars_dict[self.id_var],
            repeats=np.array(
                [len(item) for item in all_vars_dict[all_vars[0]]]
            ) * self.k,
            axis=0)

        if self._verbose is True:
            print(f"{step_cntr}. Concatenate data")
            step_cntr += 1
        for var in all_vars:
            all_vars_dict[var] = np.array(list(
                itertools.chain.from_iterable(all_vars_dict[var])
            )).flatten()

        if self._verbose is True:
            print(f"{step_cntr}. Tokenize data")
            step_cntr += 1
        _breaks = np.linspace(
            start=np.min(all_vars_dict[self._distance_var]),
            stop=np.max(all_vars_dict[self._distance_var]),
            num=self.n_bin + 1
        )
        _breaks[0] = -1
        _breaks[-1] = np.inf
        self._breaks = _breaks

        all_vars_dict[self._token_var] = self.__tokenizer(
            vec_symbol=all_vars_dict[self._symbols_var],
            vec_distance=all_vars_dict[self._distance_var],
        )

        if self._verbose is True:
            print(f"{step_cntr}. Construct dataframe")
            step_cntr += 1
        self.knn_attr = pd.DataFrame(all_vars_dict)
        return self.knn_attr
