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

def read_SMRT(path, tautomer_enumerator:Union[None, rdMolStandardize.TautomerEnumerator]=None):
    SMRT_df = pd.read_csv(path, sep=";")
    data_list  = []
    mol_list   = []
    n_embd_err = 0
    n_pars_err = 0
    for index, row in tqdm(SMRT_df.iterrows(), total = SMRT_df.shape[0], desc="Read file"):
        inchi = row["inchi"]
        mol = Chem.MolFromInchi(inchi=inchi)
        if isinstance(mol, Chem.rdchem.Mol):
            flg = AllChem.EmbedMolecule(mol)
            if flg == 0:
                if tautomer_enumerator is not None:
                    mol = tautomer_enumerator.Canonicalize(mol)
                    tautomers = [mol for mol in tautomer_enumerator.Enumerate(mol)]

                    for i, tautomer in enumerate(tautomers):
                        if i == 0:
                            data_list.append(Data(cid=row["pubchem"], rt=row["rt"], is_tautomer = False))
                        else:
                            data_list.append(Data(cid=row["pubchem"], rt=row["rt"], is_tautomer = True))
                        mol_list.append(tautomer)
                else:
                    data_list.append(Data(cid=row["pubchem"], rt=row["rt"], is_tautomer = False))
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

class SMRT(InMemoryDataset):
    def __init__(self, root, split="train", transform=None, pre_transform=None, pre_filter=None, n_jobs:int=1, splitter=None, tautomer:int=-1):
        self.n_jobs = n_jobs
        self.tautomer = tautomer
        
        if splitter is None:
            print("Using default splitter")
            self.splitter = ART.DataSplitter.CidRandomSplitter()
        else:
            self.splitter = splitter

        super(SMRT, self).__init__(root, transform, pre_transform, pre_filter)
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
        return ['SMRT.csv']
    
    @property
    def processed_file_names(self):
        return ['SMRT_train.pt', 'SMRT_validate.pt', 'SMRT_test.pt']
    
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
        data_list, mol_list = read_SMRT(self.raw_paths[0], enumerator)

        print("  > 2. Filter data")
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        print(f"  \t- # Data({len(data_list)})")
        print(f"  \t- n_jobs: {self.n_jobs:d}, max tautomer: {self.tautomer:d}")
        
        if self.pre_transform is not None:
            if self.n_jobs == 1:
                print("  > 3. Generate Atom and Bond features, Compute sequentially")
                data_list = [self.pre_transform(data) for data in data_list]
            else:
                print("  > 3. Generate Atom and Bond features, Compute parallely")
                data_list = Parallel(n_jobs=self.n_jobs)(delayed(self.pre_transform)(mol, data) for mol, data in zip(mol_list, data_list))
        
        data_list = [data for data in data_list if data is not None]

        print("  > 4. Split data")
        train_set, valid_set, test_set = self.splitter(data_list)

        print("  > 5. Save data")
        train_data , train_slices = self.collate(train_set)
        valid_data , valid_slices = self.collate(valid_set)
        test_data  , test_slices  = self.collate(test_set)
        
        torch.save((train_data, train_slices), self.processed_paths[0])
        torch.save((valid_data, valid_slices), self.processed_paths[1])
        torch.save((test_data, test_slices), self.processed_paths[2])

