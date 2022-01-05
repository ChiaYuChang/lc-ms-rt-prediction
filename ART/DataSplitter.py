import itertools
import re
import pandas as pd
import numpy as np
import random

from collections import defaultdict
from torch_geometric.data import Data
from typing import Callable, Union
from ART.funcs import gen_random_str

class DataSplitter:
    def __init__(
            self, frac_train: float = 0.8, frac_valid: float = 0.1,
            frac_test: Union[float, None] = None, seed: Union[int, None] = None
            ) -> None:

        if frac_test is None:
            frac_test = 1.0 - frac_train - frac_valid
            if frac_test < 0:
                raise ValueError("When the frac_test parameter is \
                    not given, the sum of frac_train and \
                    frac_valid should be less than 1.0.")

        frac_sum = frac_train + frac_valid + frac_test
        self.frac_train, self.frac_valid, self.frac_test = (
            frac_train/frac_sum, frac_valid/frac_sum, frac_test/frac_sum
        )

        if seed is None:
            self._seed = random.randrange(1e8)
        else:
            self._seed = seed

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, seed: Union[int, None] = None):
        if seed is None:
            new_seed = random.randint(0, 1e8)
        else:
            new_seed = seed
        print(f'Setting {new_seed} as new random seed.')
        self._seed = new_seed

    def __name__(self) -> str:
        return "DataSplitter"

    def process(self, data_list):
        raise NotImplementedError

    def calc_group_size(self, n_data):
        train_n = int(np.ceil(self.frac_train * n_data))
        valid_n = int(np.floor(self.frac_valid * n_data))

        # should contains at least 1 sample in validation set
        if valid_n == 0:
            valid_n += 1
            train_n -= 1

        # should contains at least 1 sample in test set
        if n_data - train_n - valid_n == 0:
            train_n -= 1
        test_n = int(n_data - train_n - valid_n)
        return (train_n, valid_n, test_n)

    def __call__(self, data_list):
        train_inds, valid_inds, test_inds = self.process(data_list)
        train_set = [data_list[i] for i in train_inds]
        validate_set = [data_list[i] for i in valid_inds]
        test_set = [data_list[i] for i in test_inds]
        return (train_set, validate_set, test_set)


class RandomSplitter(DataSplitter):
    def __init__(
            self, frac_train: float = 0.8, frac_valid: float = 0.1,
            frac_test: Union[float, None] = 0.1, seed: Union[int, None] = None
            ) -> None:
        super(RandomSplitter, self).__init__(
            frac_train=frac_train, frac_valid=frac_valid,
            frac_test=frac_test, seed=seed
        )

    def __name__(self) -> str:
        return "RandomSplitter"

    def process(self, data_list):
        random.seed(self.seed)
        train_n, valid_n, test_n = self.calc_group_size(n_data=len(data_list))

        # shuffe index
        group = [0] * train_n + [1] * valid_n + [2] * test_n
        random.shuffle(group)

        train_inds = []
        valid_inds = []
        test_inds = []

        for i, g in enumerate(group):
            if g == 0:
                train_inds += [int(i)]
            elif g == 1:
                valid_inds += [int(i)]
            else:
                test_inds += [int(i)]

        return (train_inds, valid_inds, test_inds)


class CidRandomSplitter(DataSplitter):
    def __init__(
            self, frac_train: float = 0.8,
            frac_valid: float = 0.1,
            frac_test: Union[float, None] = 0.1,
            seed: Union[int, None] = None,
            get_cid: Union[Callable[[Data], str], None] = None
            ) -> None:

        super(CidRandomSplitter, self).__init__(
            frac_train=frac_train,
            frac_valid=frac_valid,
            frac_test=frac_test,
            seed=seed
        )
        if get_cid is None:
            self._get_cid = lambda data: data.sup["cid"]
        else:
            self._get_cid = get_cid

    def __name__(self) -> str:
        return "CidRandomSplitter"

    def process(self, data_list):
        random.seed(self.seed)
        # random.shuffle(data_list)
        cid = pd.DataFrame({"cid": np.array(
            [self._get_cid(data) for data in data_list])}
        )
        cid["id"] = range(0, len(data_list))
        cid = pd.DataFrame(cid.groupby(by="cid")["id"].apply(list))

        train_n, valid_n, test_n = self.calc_group_size(n_data=cid.shape[0])

        group = np.array([0] * train_n + [1] * valid_n + [2] * test_n)
        np.random.shuffle(group)
        cid["group"] = group
        cid = cid.groupby("group")["id"].apply(itertools.chain.from_iterable)

        train_inds = cid[0]
        valid_inds = cid[1]
        test_inds = cid[2]

        return (train_inds, valid_inds, test_inds)


class ScaffoldSplitter(DataSplitter):
    def __init__(
            self, frac_train: float = 0.8,
            frac_valid: float = 0.1,
            frac_test: Union[float, None] = 0.1,
            seed: Union[int, None] = None,
            get_scaffold: Union[Callable[[Data], str], None] = None
            ) -> None:

        super(ScaffoldSplitter, self).__init__(
            frac_train=frac_train,
            frac_valid=frac_valid,
            frac_test=frac_test,
            seed=seed
        )

        if get_scaffold is None:
            def default_get_scaffold(data):
                s = data.sup["scaffold"]
                if s == "":
                    s = gen_random_str(20)
                return s
            self._get_scaffold = default_get_scaffold
        else:
            self._get_scaffold = get_scaffold

    def __name__(self) -> str:
        return "ScaffoldSplitter"

    def process(self, data_list):
        random.seed(self.seed)

        train_n, valid_n, _ = self.calc_group_size(n_data=len(data_list))
        train_cutoff = train_n
        valid_cutoff = train_n + valid_n
        train_inds = []
        valid_inds = []
        test_inds = []

        scaffolds = defaultdict(lambda: [])
        for idx, data in enumerate(data_list):
            scaffolds[self._get_scaffold(data)] += [idx]

        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                scaffolds.items(),
                key=lambda x: (len(x[1]), x[1][0]),
                reverse=True
            )
        ]

        for scaffold_set in scaffold_sets:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                if (len(train_inds) + len(valid_inds)
                        + len(scaffold_set) > valid_cutoff):
                    test_inds += scaffold_set
                else:
                    valid_inds += scaffold_set
            else:
                train_inds += scaffold_set
        return (train_inds, valid_inds, test_inds)


if __name__ == '__main__':
    splitter = RandomSplitter(frac_train=0.7, frac_valid=0.15, seed=2132)
    print("Fraction of each set")
    print(f"  > Train      : {splitter.frac_train:5.3f}")
    print(f"  > Validation : {splitter.frac_valid:5.3f}")
    print(f"  > Test       : {splitter.frac_test:5.3f}")
