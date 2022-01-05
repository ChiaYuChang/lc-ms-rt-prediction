# %%
from ART.DataSplitter import DataSplitter, RandomSplitter
from ART.IO import MongodReader

import json
import torch
from torch_geometric.data import InMemoryDataset

# %%
class RTData(InMemoryDataset):
    def __init__(self, root, profile_name:str="profile.json", split:str="train",
    transform=None, pre_transform=None, 
    pre_filter=None, splitter:DataSplitter= RandomSplitter):
        """
        splitter: A DataSplitter object that use to split data into different sets
        """
        self.splitter = splitter
        self.profile_name = profile_name
        self.profile_mongo = None
        self.profile_rt = None
        self.profile_mol_attrs = None
        super().__init__(root, transform, pre_transform, pre_filter)
        
        if split == "train":
            path = self.processed_paths[0]
        elif split == "valid":
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        else:
            raise ValueError(f"split variables should be either 'train', 'validate', or 'test'")
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return self.profile_name
    
    @property
    def processed_file_names(self):
        return ["train.pt", "valid.pt", "test.pt"]
    
    def process(self) -> None:
        step = 1
        print(f"  > {step}. Read profile")
        step += 1
        with open("/".join([self.root, "raw", self.raw_file_names]), "r") as json_file:
            profile = json.load(fp = json_file)
            self.profile_mongo     = profile["mongo"]
            self.profile_rt        = profile["rt"]
            self.profile_mol_attrs = profile["attrs"]

        print(f"  > {step}. Connect to mongod")
        step += 1
        self.mgMlAttrRdr = MongodReader(
            host     = self.profile_mongo["host"],
            port     = self.profile_mongo["port"],
            db       = self.profile_mongo["db"],
            col_rt   = self.profile_mongo["collection_rt_data"],
            col_attr = self.profile_mongo["collection_attrs"]
        )
        
        print(f"  > {step}. Read data")
        step += 1
        
        profile_rt = self.profile_rt
        self.mgMlAttrRdr.read_rt(
            query        = profile_rt["query"],
            exclude_cols = profile_rt["exclude_cols"]
        )
        
        profile_attrs = self.profile_mol_attrs
        self.mgMlAttrRdr.read_mol_attrs(
            limit         = profile_attrs["limit"],
            n_mol         = profile_attrs["n_mol"],
            add_cid_by_tt = profile_attrs["mol"]["add_cid_by_tt"],
            knn_to_vec    = profile_attrs["node"]["knn_to_vec"],
            k             = profile_attrs["node"]["k"],
            max_distance  = profile_attrs["node"]["max_distance"]
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
        
        print(f"  > {step}. Split data")
        step += 1
        train_set, valid_set, test_set = self.splitter(data_list)

        print(f"  > {step}. Save data")
        step += 1
        train_data, train_slices = self.collate(train_set)
        valid_data, valid_slices = self.collate(valid_set)
        test_data , test_slices  = self.collate(test_set)
        
        torch.save((train_data, train_slices), self.processed_paths[0])
        torch.save((valid_data, valid_slices), self.processed_paths[1])
        torch.save((test_data, test_slices), self.processed_paths[2])
        return None

# %%
if __name__ == '__main__':
    RTData(root="./SMRT", profile_name="SMRT.json")

# %%
