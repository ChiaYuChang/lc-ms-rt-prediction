import pandas as pd
import numpy as np
import torch
import itertools

from ART.Featurizer.Features import mol_num_atom, Feature
from ART.Featurizer.FeatureSet import FeatureSet
from ART.FileReaders import MolRecord, MolReader
from ART.funcs import split_list, calc_dist_bw_node, np_one_hot
from ART.funcs import calc_knn_graph, calc_radius_graph
from pathos.multiprocessing import ProcessingPool
from rdkit.Chem.rdchem import Mol, Atom, Bond
from torch_geometric.data import Data
from torch.nn.functional import one_hot
from tqdm import tqdm
from typing import Dict, List, Union, Dict


class Featurizer():
    def __init__(
            self, feature_set: FeatureSet,
            include_coordinates: bool = True,
            knn: int = -1, use_np: bool = True,
            radius: float = -1.0) -> None:
        if len(feature_set.mol) < 1:
            self._mol_feature_list = [mol_num_atom]
        else:
            self._mol_feature_list = feature_set.mol
        
        self._include_coordinates = include_coordinates
        if self._include_coordinates:
            self._knn = knn
            self._radius = radius
        else:
            self._knn = -1
            self._radius = -1.0
        self._use_np = use_np
        self._undirected = True
        self._atom_feature_list = feature_set.atom
        self._bond_feature_list = feature_set.bond
        self._sup_feature_list = feature_set.sup
        self._num_feature = len(self.mol_feature_list) +\
            len(self.atom_feature_list) +\
            len(self.bond_feature_list) +\
            len(self.sup_feature_list)
    
    @property
    def use_np(self) -> bool:
        return self._use_np

    @property
    def knn(self) -> int:
        return self._knn

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def undirected(self) -> bool:
        return self._undirected

    @property
    def include_coordinates(self) -> bool:
        return self._include_coordinates

    @property
    def atom_feature_list(self) -> Union[List[Feature], None]:
        return self._atom_feature_list

    @property
    def bond_feature_list(self) -> Union[List[Feature], None]:
        return self._bond_feature_list

    @property
    def mol_feature_list(self) -> Union[List[Feature], None]:
        return self._mol_feature_list

    @property
    def sup_feature_list(self) -> Union[List[Feature], None]:
        return self._sup_feature_list

    @property
    def feature_list(self):
        features = []

        # print("Mol Features:")
        if self.mol_feature_list is not None:
            features.extend(self.mol_feature_list)

        # print("Atom Features:")
        if self.atom_feature_list is not None:
            features.extend(self.atom_feature_list)

        # print("Bond Features:")
        if self.bond_feature_list is not None:
            features.extend(self.bond_feature_list)

        # print("Supplementary Features:")
        if self.sup_feature_list is not None:
            features.extend(self.sup_feature_list)

        return features

    def _gen_features(self, obj, features_list) -> Dict:
        features_dict = {}
        for f in features_list:
            if f.mapping is None:
                if self.use_np:
                    features_dict[f.name] = np.array(
                        [f.func(obj)], dtype=np.float32)
                else:
                    features_dict[f.name] = torch.tensor(
                        [f.func(obj)], dtype=torch.float32)
            else:
                if self.use_np:
                    features_dict[f.name] = np_one_hot(
                        np.array(f.mapping[f.func(obj)]),
                        num_classes=f.num_level
                    )
                else:
                    features_dict[f.name] = one_hot(
                        torch.tensor(f.mapping[f.func(obj)]),
                        num_classes=f.num_level
                    ).type(torch.float32)
        return features_dict

    def gen_mol_features(self, mol: Mol) -> Dict:
        return self._gen_features(
            obj=mol, features_list=self.mol_feature_list)

    def gen_atom_features(self, atom: Atom) -> Dict:
        return self._gen_features(
            obj=atom, features_list=self.atom_feature_list)

    def gen_bond_features(self, bond: Bond) -> Dict:
        return self._gen_features(
            obj=bond, features_list=self.bond_feature_list)

    def gen_sup_features(self, mol: Mol) -> Dict:
        sup_dict = {}
        for f in self.sup_feature_list:
            sup_dict[f.name] = f.func(mol)
        return sup_dict

    def gen_edge_index(self, mol: Mol, undirected: bool = True):
        if self._use_np:
            edge_index = np.empty((2, mol.GetNumBonds()), dtype=np.long)
            for i, bond in enumerate(mol.GetBonds()):
                edge_index[:, i] = np.array([
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx()
                ])
            if undirected:
                edge_index_inv = np.empty(edge_index.shape, dtype=np.long)
                edge_index_inv[1, :] = edge_index[0, :]
                edge_index_inv[0, :] = edge_index[1, :]
                edge_index = np.concatenate((edge_index, edge_index_inv), axis=1)
        else:
            edge_index = torch.empty((2, mol.GetNumBonds()), dtype=torch.long)
        
            for i, bond in enumerate(mol.GetBonds()):
                edge_index[:, i] = torch.tensor([
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx()
                ])
            if undirected:
                edge_index_inv = torch.empty(edge_index.shape, dtype=torch.long)
                edge_index_inv[1, :] = edge_index[0, :]
                edge_index_inv[0, :] = edge_index[1, :]
                edge_index = torch.cat((edge_index, edge_index_inv), dim=1)
        
        return(edge_index)

    def gen_knn_edge_index(self, data: Data):
        return calc_knn_graph(data=data, k=self._knn)

    def gen_radius_edge_index(self, data: Data):
        return calc_radius_graph(data=data, r=self._radius)

    def gen_3d_coordinates(self, mol: Mol):
        conformer = mol.GetConformer()
        num_atom = mol.GetNumAtoms()
        
        if self._use_np:
            coordinates = np.empty((num_atom, 3), dtype=np.float32)
            for i in range(num_atom):
                coordinates[i, :] = np.array(conformer.GetAtomPosition(i))
        else:
            coordinates = torch.empty((num_atom, 3), dtype=torch.float32)
            for i in range(num_atom):
                coordinates[i, :] = torch.tensor(conformer.GetAtomPosition(i))
        
        return(coordinates)

    def gen_dist_bw_node(self, data: Data, type_edge_index: str):
        return calc_dist_bw_node(data=data, edge_index=data[type_edge_index])

    def __call__(
            self, mol_record: Union[MolRecord, List[MolRecord], MolReader]
            ) -> Union[Data, List[Data]]:
        if isinstance(mol_record, MolRecord):
            return self.featurize(mol_record)
        elif isinstance(mol_record, list) or isinstance(mol_record, MolReader):
            data_list = [None] * len(mol_record)
            for i, r in enumerate(tqdm(mol_record, desc="featurizing", total=len(mol_record))):
                if isinstance(r, MolRecord):
                    if r.rt > 0:
                        data_list[i] = self.featurize(r)
                    else:
                        data_list[i] = None
                else:
                    raise TypeError
            return data_list
        else:
            raise TypeError

    def featurize(self, mol_record: MolRecord) -> Data:
        mol = mol_record.mol

        features_dict = {}
        if self.include_coordinates:
            features_dict["pos"] = self.gen_3d_coordinates(mol=mol)

        mol_features = self.gen_mol_features(mol=mol)
        
        if self._use_np:
            features_dict["graph_attr"] = np.concatenate(
                [mol_features[item.name] for item in self.mol_feature_list])
            # features_dict["graph_attr"] =  [mol_features[item.name] for item in self.mol_feature_list]
        else:
            features_dict["graph_attr"] = torch.cat(
                [mol_features[item.name] for item in self.mol_feature_list])

        features_dict["sup"] = {**mol_record.supplementary, **self.gen_sup_features(mol=mol)}

        atom_features = [None] * mol.GetNumAtoms()
        for idx, atom in enumerate(mol.GetAtoms()):
            atom_features[idx] = self.gen_atom_features(atom=atom)
        # Encode Atom Feature
        atom_features = pd.DataFrame.from_records(atom_features)
        atom_features = atom_features[[item.name for item in self.atom_feature_list]]
        atom_features = atom_features.to_records(index=False)
        
        if self._use_np:
            atom_features = np.stack(
                [np.concatenate(tuple(r)) for r in atom_features]
            )
            # atom_features = [r for r in atom_features]
        else:
            atom_features = torch.stack(
                tuple(torch.cat(tuple(r)) for r in atom_features)
            )
        features_dict["node_attr"] = atom_features

        bond_features = [None] * mol.GetNumBonds()
        for idx, bond in enumerate(mol.GetBonds()):
            bond_features[idx] = self.gen_bond_features(bond=bond)
        # Encode Bond Feature
        bond_features = pd.DataFrame.from_records(bond_features)
        bond_features = bond_features[[item.name for item in self.bond_feature_list]]
        bond_features = bond_features.to_records(index=False)
        
        if self._use_np:
            bond_features = np.stack(
                [np.concatenate(tuple(r)) for r in bond_features]
            )
            # bond_features = [r for r in bond_features]
        else:
            bond_features = torch.stack(
                tuple(torch.cat(tuple(r)) for r in bond_features)
            )

        if self.undirected:
            if self._use_np:
                bond_features = np.concatenate([bond_features, bond_features])
            else:
                bond_features = torch.cat([bond_features, bond_features])

        features_dict["edge_attr"] = bond_features
        features_dict["edge_index"] = self.gen_edge_index(
            mol=mol, undirected=self.undirected)

        if not(features_dict["edge_attr"].shape[0] == features_dict["edge_index"].shape[1]):
            raise RuntimeError

        data = Data.from_dict(features_dict)

        if self._knn > 0:
            data["knn_edge_index"] = self.gen_knn_edge_index(data=data)
            data["knn_edge_attr"] = self.gen_dist_bw_node(
                data=data, type_edge_index="knn_edge_index")

        if self._radius > 0:
            data["radius_edge_index"] = self.gen_radius_edge_index(data=data)
            data["radius_edge_attr"] = self.gen_dist_bw_node(
                data=data, type_edge_index="radius_edge_index")

        if self._use_np:
            data["y"] = np.array([mol_record.rt])
        else:
            data["y"] = torch.tensor([mol_record.rt])

        return data
    
    def __len__(self) -> int:
        return self._num_feature

    def __repr__(self) -> str:
        return f"Featurizer ({self._num_feature})"

class ParallelFeaturizer():
    def __init__(
            self, featurizer: Featurizer, n_jobs: int = 2,
            rm_None: bool = False) -> None:
        self.n_jobs = n_jobs
        self._data_list = None
        self._featurizer = featurizer
        self._rm_None = rm_None

    def __call__(
            self, file_reader: MolReader, pool) -> List[Data]:
        record_list = self.split_list(x=file_reader, n=self.n_jobs)
        data_list = pool.map(
            self._featurizer,
            record_list
        )
        data_list = list(itertools.chain.from_iterable(data_list))
        if self.rm_None:
            data_list = [item for item in data_list if item is not None]
        self._data_list = data_list
        return self._data_list

    def split_list(self, x: List, n: int) -> List:
        return split_list(x=x, n=n, return_idxs=False)

    @property
    def data_list(self):
        return self._data_list

    @property
    def featurizer(self):
        return self._featurizer

    @property
    def rm_None(self):
        return self._rm_None
