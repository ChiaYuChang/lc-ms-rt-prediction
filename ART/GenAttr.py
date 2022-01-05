import functools
import pandas as pd
import torch

from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Crippen
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from torch_geometric.data import Data
from tqdm import tqdm
from typing import Union


class GenAttrs:
    def __init__(self) -> None:
        pass

    def __call__(
            self, mol: Chem.rdchem.Mol,
            data: Union[Data, None] = None
            ) -> Data:
        if data is None:
            data = self.empty()
        return data

    def empty(self) -> Data:
        return Data()


class GenMolAttrs(GenAttrs):
    def __init__(self) -> None:
        super(GenMolAttrs, self).__init__()
        
        def grouping(
                x: int, lb: int, lb_label: int, 
                ub: int, ub_label: int) -> int:
            if x <= lb:
                return lb_label
            elif x > ub:
                return ub_label
            else:
                return x

        self.n_hba_dict = defaultdict(lambda: 1)
        for n_hba in range(0, 20):
            self.n_hba_dict[n_hba] = grouping(
                x=n_hba, lb=1, lb_label=0, ub=9, ub_label=1
            )
        
        self.n_hbd_dict = defaultdict(lambda: 4)
        for n_hbd in range(0, 14):
            self.n_hbd_dict[n_hbd] = grouping(
                x=n_hbd, lb=0, lb_label=0, ub=3, ub_label=4
            )

        self.n_ring_dict = defaultdict(lambda: 7)
        for n_ring in range(0, 12):
            self.n_ring_dict[n_ring] = grouping(
                x=n_ring, lb=0, lb_label=0, ub=6, ub_label=7
            )

        self.n_aromatic_ring_dict = defaultdict(lambda: 6)
        for n_a_ring in range(0, 8):
            self.n_aromatic_ring_dict[n_a_ring] = grouping(
                x=n_a_ring, lb=0, lb_label=0, ub=5, ub_label=6
            )

        self.n_aliphatic_ring_dict = defaultdict(lambda: 4)
        for n_a_ring in range(0, 9):
            self.n_aliphatic_ring_dict[n_a_ring] = grouping(
                x=n_a_ring, lb=0, lb_label=0, ub=3, ub_label=4
            )

    def __call__(
            self, mol: Chem.rdchem.Mol, data: Union[Data, None] = None
            ) -> Data:
        if data is None:
            data = super().__call__(mol, data=data)
        data["formula"] = rdMolDescriptors.CalcMolFormula(mol)
        data["mw"] = rdMolDescriptors.CalcExactMolWt(mol)
        data["volume"] = AllChem.ComputeMolVolume(mol)
        data["smiles"] = AllChem.MolToSmiles(mol)
        data["num_hba"] = self.n_hba_dict[rdMolDescriptors.CalcNumHBA(mol)]
        data["num_hbd"] = self.n_hbd_dict[rdMolDescriptors.CalcNumHBD(mol)]
        data["num_ring"] = self.n_ring_dict[rdMolDescriptors.CalcNumRings(mol)]
        data["num_aromatic_ring"] = self.n_aromatic_ring_dict[
            rdMolDescriptors.CalcNumAromaticRings(mol)
        ]
        data["num_aliphatic_ring"] = self.n_aliphatic_ring_dict[
            rdMolDescriptors.CalcNumAliphaticRings(mol)
        ]
        data["molLogP"] = Crippen.MolLogP(mol)
        data["scaffold"] = MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        )
        data["num_nodes"] = mol.GetNumAtoms()
        return(data)

    def empty(self, data: Data) -> Data:
        data["formula"] = ""
        data["mw"] = -1.0
        data["volume"] = -1.0
        data["smiles"] = None
        data["num_hba"] = -1
        data["num_hbd"] = -1
        data["num_ring"] = -1
        data["num_aromatic_ring"] = -1
        data["num_aliphatic_ring"] = -1
        data["molLogP"] = -1.0
        data["scaffold"] = ""
        data["num_nodes"] = -1
        return Data


class GenAtomAttrs(GenAttrs):
    def __init__(self, k:int=5) -> None:
        super(GenAtomAttrs, self).__init__()
        symbols = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si"]
        self.symbols_dict = defaultdict(lambda: 0)
        for i, s in zip(range(len(symbols)), symbols):
            self.symbols_dict[s] = int(i)+1
        
        hybridization = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3
        ]
        self.hybridization_dict = defaultdict(lambda: 0)
        for i, h in zip(range(len(hybridization)), hybridization):
            self.hybridization_dict[h] = int(i)+1
    
        degree = range(1, 5)
        self.degree_dict = defaultdict(lambda: 0)
        for i, d in zip(range(len(degree)), degree):
            self.degree_dict[d] = int(i)+1

        n_hs = range(1, 5)
        self.n_hs_dict = defaultdict(lambda: 0)
        for i, h in zip(range(len(n_hs)), n_hs):
            self.n_hs_dict[h] = int(i)+1

        formal_charge = [-1, 0, 1]
        self.formal_charge_dict = defaultdict(lambda: 0)
        for i, c in zip(range(len(formal_charge)), formal_charge):
            self.formal_charge_dict[c] = int(i)+1
        
        self.k = k
    
    def __call__(self, mol: Chem.rdchem.Mol, data: Union[Data, None]=None) -> Data:
        if data is None:
            data = super().__call__(mol, data=data)
        
        num_nodes = mol.GetNumAtoms()
        atom = pd.DataFrame({
            "index": range(0, num_nodes),
            "atom": [atom for atom in mol.GetAtoms()], 
        })
            
        atom.set_index("index", inplace=True)
        atom["symbol"]        = [self.symbols_dict[atom.GetSymbol()] for atom in atom.atom]
        atom["hybridization"] = [self.hybridization_dict[atom.GetHybridization()] for atom in atom.atom]
        atom["degree"]        = [self.degree_dict[atom.GetDegree()] for atom in atom.atom]
        atom["num_hs"]        = [self.n_hs_dict[atom.GetTotalNumHs()] for atom in atom.atom]
        atom["formal_charge"] = [self.formal_charge_dict[atom.GetFormalCharge()] for atom in atom.atom]
        atom["aromaticity"]   = [1 if atom.GetIsAromatic() else 0 for atom in atom.atom]
        atom["in_ring"]       = [1 if atom.IsInRing() else 0 for atom in atom.atom]

        if self.k > 0:
            dist = pd.DataFrame(Chem.Get3DDistanceMatrix(mol))
            dist["from_atom_index"] = range(mol.GetNumAtoms())
            dist = dist.melt("from_atom_index", var_name = "to_atom_index", value_name = "distance")
            dist = dist[dist["from_atom_index"] != dist["to_atom_index"]]
            dist = dist.sort_values(["from_atom_index", "distance"]).groupby("from_atom_index").apply(lambda x: x.head(self.k))
            dist.reset_index(drop=True, inplace=True)
            dist = dist.set_index("to_atom_index")
            dist = dist.join(atom[['symbol']])
            dist.rename(columns={"symbol":"to_atom_one_hot"}, inplace=True)
            dist.reset_index(drop=True, inplace=True)
            dist["distance"] = [[dist] for dist in dist["distance"]]
            dist["to_atom_one_hot"] = [[atm] for atm in dist["to_atom_one_hot"]]
            dist = dist.groupby("from_atom_index")[["to_atom_one_hot", "distance"]].agg(lambda x: functools.reduce(lambda x, y: x+y, x))
            atom = atom.join(dist)
        
        data["atom_symbol"] = torch.tensor(atom["symbol"])
        data["atom_hybridization"]= torch.tensor(atom["hybridization"])
        data["atom_degree"] = torch.tensor(atom["degree"])
        data["atom_num_hs"] = torch.tensor(atom["num_hs"])
        data["atom_formal_charge"]= torch.tensor(atom["formal_charge"])
        data["atom_aromaticity"] = torch.tensor(atom["aromaticity"])
        data["atom_in_ring"] = torch.tensor(atom["in_ring"])
        
        if self.k  > 0:
            data["atom_knn_atom"] = torch.tensor(atom["to_atom_one_hot"])
        data["k"] = self.k
        return data


class GenBondAttrs(GenAttrs):
    def __init__(self) -> None:
        super(GenBondAttrs, self).__init__()
        bondtype = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]
        self.bondtype_dict = defaultdict(lambda: 0)
        for i, b in zip(range(len(bondtype)), bondtype):
            self.bondtype_dict[b] = int(i)+1

        stereos = [
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ
        ]
        self.stereos_dict = defaultdict(lambda: 0)
        for i, s in zip(range(len(stereos)), stereos):
            self.stereos_dict[s] = int(i)+1

    def __call__(self, mol: Chem.rdchem.Mol, data: Union[Data, None]) -> Data:
        if data is None:
            data = super().__call__(mol, data=data)

        num_edges = mol.GetNumBonds()
        edge = pd.DataFrame({
            "index": range(0, num_edges),
            "bond": [bond for bond in mol.GetBonds()], 
        })
        # print("Extracting bond features")
        edge["bondtype"]    = [self.bondtype_dict[bond.GetBondType()] for bond in edge.bond]
        edge["conjugation"] = [1 if bond.GetIsConjugated() else 0 for bond in edge.bond]
        edge["in_ring"]     = [1 if bond.IsInRing() else 0 for bond in edge.bond]
        edge["stereos"]     = [self.stereos_dict[bond.GetStereo()] for bond in edge.bond]

        bond_begin_atom_idx = [bond.GetBeginAtomIdx() for bond in edge.bond]
        bond_end_atom_idx   = [bond.GetEndAtomIdx() for bond in edge.bond]

        data["bond_bondtype"]    = torch.cat([torch.tensor(edge["bondtype"]), torch.tensor(edge["bondtype"])])
        data["bond_conjugation"] = torch.cat([torch.tensor(edge["conjugation"]), torch.tensor(edge["conjugation"])]) 
        data["bond_in_ring"]     = torch.cat([torch.tensor(edge["in_ring"]), torch.tensor(edge["in_ring"])]) 
        data["bond_stereos"]     = torch.cat([torch.tensor(edge["stereos"]), torch.tensor(edge["stereos"])]) 

        data["edge_index"] = torch.stack([
            torch.tensor(bond_begin_atom_idx + bond_end_atom_idx),
            torch.tensor(bond_end_atom_idx + bond_begin_atom_idx)
        ])
        return data


class AttrsGenPipeline(GenAttrs):
    def __init__(self, fun_lst) -> None:
        super(GenAttrs, self).__init__()
        self.fun_lst = fun_lst
    
    def __call__(self, mol: Chem.rdchem.Mol, data: Union[Data, None]=None) -> Data:
        if data is None:
            data = super().__call__(mol, data=data)
        for fun in self.fun_lst:
            data = fun(mol, data)
        return data

# Testing code
if __name__ == '__main__':
    gen_mol_attr  = GenMolAttrs()
    gen_atom_attr = GenAtomAttrs(k=3)
    gen_bond_attr = GenBondAttrs()
    pre_transform = AttrsGenPipeline(fun_lst=[gen_mol_attr, gen_atom_attr, gen_bond_attr])

    cid   = [3505, 2159, 1340]
    rt    = [687.8, 590.7, 583.6] 
    inchi = [
        "InChI=1S/C19H25Cl2N3O3/c1-27-19(26)23-8-9-24(15(13-23)12-22-6-2-3-7-22)18(25)11-14-4-5-16(20)17(21)10-14/h4-5,10,15H,2-3,6-9,11-13H2,1H3/t15-/m1/s1",
        "InChI=1S/C17H27N3O4S/c1-4-20-8-6-7-12(20)11-19-17(21)13-9-16(25(22,23)5-2)14(18)10-15(13)24-3/h9-10,12H,4-8,11,18H2,1-3H3,(H,19,21)/t12-/m1/s1",
        "InChI=1S/C9H7NO2/c11-8-3-1-2-7-6(8)4-5-10-9(7)12/h1-5,11H,(H,10,12)"
    ]
    
    mol_list  = []
    data_list = []

    for c, r, i in zip(cid, rt, inchi):
        mol = Chem.MolFromInchi(inchi=i)
        if AllChem.EmbedMolecule(mol) == 0:
            mol_list.append(mol)
            data_list.append(Data(cid=c, rt=r, inchi=i, phase="POS"))
    
    data_list = [pre_transform(mol, data) for mol, data in tqdm(zip(mol_list, data_list), total = len(mol_list), desc="Gen Attrs\t")]
    
    for (i, data) in enumerate(data_list):
        print("\n")
        print(f"i: {i}, SMILES: {data.smiles}")        
        print(data)