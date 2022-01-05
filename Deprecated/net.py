import pandas as pd
import deepchem as dc
import numpy as np
import itertools
import functools
import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from collections import defaultdict, namedtuple
from random import shuffle, seed
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors, Descriptors
from rdkit.Chem import rdchem
from rdkit.Chem import rdDistGeom as molDG
from rdkit.Chem import Crippen
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from joblib import Parallel, delayed

from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool, BatchNorm, LayerNorm
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax
from torch_geometric.nn.models import AttentiveFP

from torch.nn import Linear, Parameter, GRUCell
from torch.nn.init import xavier_normal_ as glorot
from torch.nn.init import zeros_ as zeros
from torch.nn.init import uniform_ as uniform

class GenAtomAndBondFeatures(object):
    def __init__(self, k):

        symbols = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si"]
        self.symbols_dict = defaultdict(lambda: [0])
        for i, s in zip(range(len(symbols)), symbols):
            self.symbols_dict[s] = [int(i)+1]
        
        hybridization = [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2
        ]

        self.hybridization_dict = defaultdict(lambda: [0])
        for i, h in zip(range(len(hybridization)), hybridization):
            self.hybridization_dict[h] = [int(i)+1]
    
        degree = range(1, 5)
        self.degree_dict = defaultdict(lambda: [0])
        for i, d in zip(range(len(degree)), degree):
            self.degree_dict[d] = [int(i)+1]

        bondtype = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]
        self.bondtype_dict = defaultdict(lambda: [0.])
        for i, b in zip(range(len(bondtype)), bondtype):
            self.bondtype_dict[b] = [int(i)+1]


        stereos = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]
        self.stereos_dict = defaultdict(lambda: [0])
        for i, s in zip(range(len(stereos)), stereos):
            self.stereos_dict[s] = [int(i)+1]

        self.k = k

    def __call__(self, data):
        try:
            # print("read Mol object")
            mol = data.mol
            
            # Generate node features
            # print("Generating atom features")
            num_nodes = mol.GetNumAtoms()
            num_edges = mol.GetNumBonds()
            atom = pd.DataFrame({
                "index": range(0, num_nodes),
                "atom": [atom for atom in mol.GetAtoms()], 
            })
            
            atom.set_index("index", inplace=True)
            atom["symbol"] = [self.symbols_dict[atom.GetSymbol()] for atom in atom.atom]
            atom["hybridization"] = [self.hybridization_dict[atom.GetHybridization()] for atom in atom.atom]
            atom["formal_charge"] = [[atom.GetFormalCharge()] for atom in atom.atom]
            atom["aromaticity"] = [[1.] if atom.GetIsAromatic() else [0.] for atom in atom.atom]
            atom["degree"] = [self.degree_dict[atom.GetDegree()] for atom in atom.atom]
            atom["num_hs"] = [[atom.GetTotalNumHs()] for atom in atom.atom]
            atom["in_ring"] = [[1.] if atom.IsInRing() else [0.] for atom in atom.atom]
            
            # print("Generating distance matrix")
            # AllChem.EmbedMolecule(mol)
            dist = pd.DataFrame(Chem.Get3DDistanceMatrix(mol))
            dist["from_atom_index"] = range(mol.GetNumAtoms())
            dist = dist.melt("from_atom_index", var_name = "to_atom_index", value_name = "distance")
            dist = dist[dist["from_atom_index"] != dist["to_atom_index"]]
            dist = dist.sort_values(["from_atom_index", "distance"]).groupby("from_atom_index").apply(lambda x: x.head(5))
            dist.reset_index(drop=True, inplace=True)
            dist = dist.set_index("to_atom_index")
            dist = dist.join(atom[['symbol']])
            dist.rename(columns={"symbol":"to_atom_one_hot"}, inplace=True)
            dist.reset_index(drop=True, inplace=True)
            dist["distance"] = [[dist] for dist in dist["distance"]]
            dist = dist.groupby("from_atom_index")[["to_atom_one_hot", "distance"]].agg(lambda x: functools.reduce(lambda x, y: x+y, x))
            atom = atom.join(dist)
            
            # print("Assigning atom features")
            data["atom_symbol"] = torch.tensor(atom["symbol"])
            data["atom_hybridization"] = torch.tensor(atom["hybridization"])
            data["atom_formal_charge"] = torch.tensor(atom["formal_charge"])
            # print("\tAssigning atom features - atom_formal_charge")
            # print("\tAssigning atom features - atom_formal_charge")
            data["atom_aromaticity"] = torch.tensor(atom["aromaticity"])
            data["atom_degree"] = torch.tensor(atom["degree"])
            data["atom_in_ring"] = torch.tensor(atom["in_ring"])
            data["atom_knn_atom"] = torch.tensor(atom["to_atom_one_hot"])
            
            # Generate edge_index
            # print("Generating edge data")
            edge = pd.DataFrame({
                "index": range(0, num_edges),
                "bond": [bond for bond in mol.GetBonds()], 
            }).set_index("index")
            
            # print("Extracting bond features")
            edge["bondtype"] = [self.bondtype_dict[bond.GetBondType()] for bond in edge.bond]
            edge["conjugation"] = [[1.] if bond.GetIsConjugated() else [0.] for bond in edge.bond]
            edge["in_ring"] = [[1.] if bond.IsInRing() else [0.] for bond in edge.bond]
            edge["stereos"] = [self.stereos_dict[bond.GetStereo()] for bond in edge.bond]
            edge["begin_atom_idx"] = [[bond.GetBeginAtomIdx()] for bond in edge.bond]
            edge["end_atom_idx"] = [[bond.GetEndAtomIdx()] for bond in edge.bond]
            
            data["bond_bondtype"] = torch.cat([torch.tensor(edge["bondtype"]), torch.tensor(edge["bondtype"])])
            data["bond_conjugation"] = torch.cat([torch.tensor(edge["conjugation"]), torch.tensor(edge["conjugation"])]) 
            data["bond_in_ring"] = torch.cat([torch.tensor(edge["in_ring"]), torch.tensor(edge["in_ring"])]) 
            data["bond_stereos"] = torch.cat([torch.tensor(edge["stereos"]), torch.tensor(edge["stereos"])]) 
            
            # print("Generating edge index")
            edge_index = torch.cat([
                torch.tensor(edge["begin_atom_idx"] + edge["end_atom_idx"]),
                torch.tensor(edge["end_atom_idx"] + edge["begin_atom_idx"])
            ]).t()
            
            # edge_attr = torch.cat([
            #     torch.tensor(edge["bondtype"] + edge["conjugation"] + edge["in_ring"] + edge["stereos"]),
            #     torch.tensor(edge["bondtype"] + edge["conjugation"] + edge["in_ring"] + edge["stereos"])
            # ])
            
            # print("Generating scaffold")
             
            
            data["edge_index"] = edge_index
            # data["edge_attr"] = edge_attr
            data["num_nodes"] = num_nodes
            return(data)
        except:
            print(data.smiles)
            return(None)