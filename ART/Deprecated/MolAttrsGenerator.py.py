import pymongo
import pandas as pd
import numpy as np
import copy
import itertools
import pymongo

from collections import defaultdict
from datetime import datetime
from joblib import Parallel, delayed
from json import dumps
from rdkit.Chem.rdchem import Mol
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Crippen
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from tqdm import tqdm
from typing import Union, Dict



def MolAttrsGenerator():
    def __init__(self, 
                mongo_host: Union[str, None] = "mongodb://localhost",
                mongo_port: Union[int, None] = "27017",
                mongo_conn_str: Union[str, None] = None
            ):
        bondtype_dict = defaultdict(lambda: "others")
        bondtype_dict[Chem.rdchem.BondType.SINGLE] = "single"
        bondtype_dict[Chem.rdchem.BondType.DOUBLE] = "double"
        bondtype_dict[Chem.rdchem.BondType.TRIPLE] = "triple"
        bondtype_dict[Chem.rdchem.BondType.AROMATIC] = "aromatic"
        self.bondtype_dict = bondtype_dict

        stereos_dict = defaultdict(lambda: "others")
        stereos_dict[Chem.rdchem.BondStereo.STEREOE] = "stereoe"
        stereos_dict[Chem.rdchem.BondStereo.STEREONONE] = "stereonone"
        stereos_dict[Chem.rdchem.BondStereo.STEREOZ] = "stereoz"
        self.stereos_dict = stereos_dict

        hybridization_dict = defaultdict(lambda: "others")
        hybridization_dict[Chem.rdchem.HybridizationType.SP] = "sp"
        hybridization_dict[Chem.rdchem.HybridizationType.SP2] = "sp2"
        hybridization_dict[Chem.rdchem.HybridizationType.SP3] = "sp3"
        self.hybridization_dict = hybridization_dict

        if mongo_conn_str is None:
            self.mongo_conn_str = mongo_conn_str
        else:
            self.mongo_conn_str = f"{mongo_host}:{mongo_port}"

    def date_time_serializer(time: datetime):
        return time.isoformat()
    
    def generate_mol_attrs(
                mol: Mol,
                mol_formula: str,
                mol_volume: float,
                rt_data_id,
                mol_wt: float,
                tautomer,
                is_tautomers: bool
            ) -> Dict:

        return({
            "_id": Chem.MolToInchiKey(mol),
            "SMILES": Chem.MolToSmiles(mol),
            "InChI": Chem.MolToInchi(mol),
            "formula": mol_formula,
            "rt_data_id": rt_data_id,
            "wt": mol_wt,
            "volume": mol_volume,
            "n_hba": rdMolDescriptors.CalcNumHBA(mol),
            "n_hbd": rdMolDescriptors.CalcNumHBD(mol),
            "n_ring": rdMolDescriptors.CalcNumRings(mol),
            "n_aromatic_ring": rdMolDescriptors.CalcNumAromaticRings(mol),
            "n_aliphatic_ring": rdMolDescriptors.CalcNumAliphaticRings(mol),
            "n_edge": mol.GetNumBonds(),
            "n_node": mol.GetNumAtoms(),
            "mLogP": Crippen.MolLogP(mol),
            "scaffold": MurckoScaffoldSmiles(mol = mol, includeChirality=False),
            "is_tautomers": is_tautomers,
            "tautomer": tautomer,
        })

