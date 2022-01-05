import torch
import ART.DataSplitter
import pandas as pd

from typing import Union
from joblib import Parallel, delayed
from rdkit.Chem.MolStandardize import rdMolStandardize
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

from typing import Union
from rdkit import Chem
from joblib import Parallel, delayed
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

def read_SELF_LAB(path, tautomer_enumerator:Union[None, rdMolStandardize.TautomerEnumerator]=None):
    SELF_df = pd.read_csv(path, sep=",")
    SELF_df["pubchem"] = SELF_df["pubchem"].astype(str)
    data_list  = []
    mol_list   = []
    n_embd_err = 0
    n_pars_err = 0
    for index, row in tqdm(SELF_df.iterrows(), total = SELF_df.shape[0], desc="Read file"):
        inchi = row["inchi"]
        mol = Chem.MolFromInchi(inchi=inchi)
        if isinstance(mol, Chem.rdchem.Mol):
            flg = AllChem.EmbedMolecule(mol)
            if flg == 0:
                if tautomer_enumerator is not None:
                    mol = tautomer_enumerator.Canonicalize(mol)
                    for tautomer in [mol for mol in tautomer_enumerator.Enumerate(mol)]:
                        data_list.append(Data(cid=row["pubchem"], rt=row["rt"], phase=row["phase"]))
                        mol_list.append(tautomer)
                else:
                    data_list.append(Data(cid=row["pubchem"], rt=row["rt"], phase=row["phase"]))
                    mol_list.append(mol)
            else:
                print(f"Embedding error\t{index:05d}: {inchi}")
                n_embd_err += 1
                next
        else:
            print(f"Parsing error\t{index:05d}: {inchi}")
            n_pars_err += 1
            next
    print(f"Number of Valid mol: {len(data_list)}, (Embedding err: {n_embd_err}, Parsing err: {n_pars_err})\n")
    return (data_list, mol_list)

class SELF_LAB(InMemoryDataset):
    def __init__(self, root, split="train", phase:str="POS", transform=None, pre_transform=None, pre_filter=None, n_jobs:int=1, splitter=None, tautomer:int=-1):
        self.n_jobs = n_jobs
        self.tautomer = tautomer
        
        if phase != "POS" and phase != "NEG":
            raise ValueError(f"phase variables should be 'POS' or 'NEG'")
        else:
            self.phase = phase

        if splitter is None:
            print("Using default splitter")
            self.splitter = ART.DataSplitter.CidRandomSplitter()
        else:
            self.splitter = splitter
        
        super(SELF_LAB, self).__init__(root, transform, pre_transform, pre_filter)
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
        return ["SELF.csv"]
    
    @property
    def processed_file_names(self):
        return [s + ".pt" for s in ["_".join(["SELF", slice, self.phase]) for slice in ["train", "validate", "test"]]]
    
    def process(self):
        if self.tautomer <= 1:
            include_tautomer = False
            max_n_tautomer   = 0
        else:
            include_tautomer = True
            max_n_tautomer   = self.tautomer

        if include_tautomer:
            enumerator = rdMolStandardize.TautomerEnumerator()
            enumerator.SetMaxTautomers(max_n_tautomer)
        else:
            enumerator = None

        print("  > 1. Read SMRT.csv")
        data_list, mol_list = read_SELF_LAB(self.raw_paths[0], enumerator)

        print("  > 2 Filter data")
        if self.pre_filter is not None:
            filter_idx = [i for i, data in tqdm(enumerate(data_list), total = len(data_list), desc="Filter\t") if self.pre_filter(data)]
            data_list = [data_list[i] for i in filter_idx]
            mol_list = [mol_list[i] for i in filter_idx]
        
        print("  > 3 Filter data by phase")
        filter_idx = [i for i, data in tqdm(enumerate(data_list), total = len(data_list), desc="Filter\t") if data.phase == self.phase]
        data_list = [data_list[i] for i in filter_idx]
        mol_list = [mol_list[i] for i in filter_idx]
        print(f"  \t- # Data({len(data_list)})")

        print(f"  \t- n_jobs: {self.n_jobs:d}, max tautomer: {self.tautomer:d}")
        if self.pre_transform is not None:
            if self.n_jobs == 1:
                print("  > 4. Generate Atom and Bond features, Compute sequentially")
                data_list = [self.pre_transform(mol, data) for mol, data in tqdm(zip(mol_list, data_list), total = len(mol_list), desc="Gen Attrs\t")]
            else:
                print("  > 4. Generate Atom and Bond features, Compute parallely")
                data_list = Parallel(n_jobs=self.n_jobs)(delayed(self.pre_transform)(mol, data) for mol, data in tqdm(zip(mol_list, data_list), total = len(mol_list), desc="Gen Attrs"))
        
        data_list = [data for data in data_list if data is not None]
        
        print("  > 5. Split data")
        train_set, valid_set, test_set = self.splitter(data_list)

        print(f"  > 6. Save data (train({len(train_set)})), (valid({len(valid_set)})), (test({len(test_set)}))")
        train_data, train_slices = self.collate(train_set)
        valid_data, valid_slices = self.collate(valid_set)
        test_data,  test_slices  = self.collate(test_set)
        
        torch.save((train_data, train_slices), self.processed_paths[0])
        torch.save((valid_data, valid_slices), self.processed_paths[1])
        torch.save((test_data, test_slices), self.processed_paths[2])
