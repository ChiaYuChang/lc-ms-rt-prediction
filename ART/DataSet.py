import torch

from ART.Data import GraphData as Data
from ART.DataSplitter import RandomSplitter, DataSplitter
# from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from typing import Callable, List, Union, Optional


class SMRT(InMemoryDataset):
    def __init__(
            self, root: str,
            data_list: Union[List[Data], None] = None,
            split: str = "train",
            transform: Optional[Callable[[Data], Data]] = None,
            pre_transform: Optional[Callable[[Data], Data]] = None,
            pre_filter: Optional[Callable[[Data], bool]] = None,
            splitter: Optional[DataSplitter] = None
            ) -> None:
        """
        splitter: A DataSplitter object that use to split data
                  into different sets
        """
        if splitter is None:
            self._splitter = RandomSplitter()
        else:
            self._splitter = splitter
        self._data_list = data_list

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
        return ["SMRT_dataset.sdf"]

    @property
    def processed_file_names(self):
        return ["train.pt", "valid.pt", "test.pt"]

    @property
    def splitter(self):
        return self._splitter

    def process(self) -> None:
        data_list = self._data_list
        
        if self.pre_filter is not None:
            print(" > Filtering data")
            data_list = [data for data in\
                tqdm(iterable=data_list, desc="Filtering data")\
                if self.pre_filter(data)]

        if self.pre_transform is not None:
            print(" > Transforming data")
            data_list = [self.pre_transform(data) for data in\
                tqdm(iterable=data_list, desc="Transforming data")]

        # split data into three set
        print(" > Splitting data")
        train_set, valid_set, test_set = self.splitter(data_list)

        # save data
        print(" > Saving data")
        train_data, train_slices = self.collate(train_set)
        torch.save((train_data, train_slices), self.processed_paths[0])

        valid_data, valid_slices = self.collate(valid_set)
        torch.save((valid_data, valid_slices), self.processed_paths[1])

        test_data, test_slices = self.collate(test_set)
        torch.save((test_data, test_slices), self.processed_paths[2])

        # Clean up data
        print(" > Cleaning up data")
        self._data_list = None
        return None


class PredRet(InMemoryDataset):
    def __init__(
            self, root: str,
            data_list: Union[List[Data], None] = None,
            split: str = "train",
            transform: Optional[Callable[[Data], Data]] = None,
            pre_transform: Optional[Callable[[Data], Data]] = None,
            pre_filter: Optional[Callable[[Data], bool]] = None,
            splitter: Optional[DataSplitter] = None
            ) -> None:
        """
        splitter: A DataSplitter object that use to split data
                  into different sets
        """
        if splitter is None:
            self._splitter = RandomSplitter()
        else:
            self._splitter = splitter
        self._data_list = data_list

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
        return None

    @property
    def raw_file_names(self):
        return ["PredRet.csv"]

    @property
    def processed_file_names(self):
        return ["train.pt", "valid.pt", "test.pt"]

    @property
    def splitter(self):
        return self._splitter

    def process(self) -> None:
        data_list = self._data_list

        if self.pre_filter is not None:
            print(" > Filtering data")
            data_list = [data for data in\
                tqdm(iterable=data_list, desc="Filtering data")\
                if self.pre_filter(data)]

        if self.pre_transform is not None:
            print(" > Transforming data")
            data_list = [self.pre_transform(data) for data in\
                tqdm(iterable=data_list, desc="Transforming data")]

        # split data into three set
        print(" > Splitting data")
        train_set, valid_set, test_set = self.splitter(data_list)

        # save data
        print(" > Saving data")
        train_data, train_slices = self.collate(train_set)
        torch.save((train_data, train_slices), self.processed_paths[0])

        valid_data, valid_slices = self.collate(valid_set)
        torch.save((valid_data, valid_slices), self.processed_paths[1])

        test_data, test_slices = self.collate(test_set)
        torch.save((test_data, test_slices), self.processed_paths[2])

        # Clean up data
        print(" > Cleaning up data")
        self._data_list = None
        return None
