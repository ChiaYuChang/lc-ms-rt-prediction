from . import Features
from .Features import Feature 
from typing import List, Union


class FeatureSet():
    __name__ = "FeatureSet"

    def __init__(
            self, 
            mol: Union[List[Feature], Feature, None] = None,
            atom: Union[List[Feature], Feature, None] = None,
            bond: Union[List[Feature], Feature, None] = None,
            sup: Union[List[Feature], Feature, None] = None
            ) -> None:
        self.mol = self._to_list(mol)
        self.atom = self._to_list(atom)
        self.bond = self._to_list(bond)
        self.sup = self._to_list(sup)

    def _to_list(self, item: Union[List[Feature], Feature, None]):
        if isinstance(item, List):
            return item
        elif isinstance(item, Feature):
            return [item]
        elif item is None:
            return []
        else:
            raise TypeError

    def __len__(self):
        return len(self.mol) + len(self.atom) + len(self.bond) + len(self.sup)

    def __repr__(self) -> str:
        return f"{self.__name__} ({self.__len__()})"

class DefaultFeatureSet(FeatureSet):
    __name__ = "DefaultFeatureSet"

    def __init__(
            self, 
            mol: Union[List[Feature], Feature, None] = None,
            atom: Union[List[Feature], Feature, None] = None,
            bond: Union[List[Feature], Feature, None] = None,
            sup: Union[List[Feature], Feature, None] = None
            ) -> None:

        super().__init__(mol=mol, atom=atom, bond=bond, sup=sup)
        
        self.mol += [
            Features.mol_num_h_acceptor,
            Features.mol_num_h_donor,
            Features.mol_num_ring,
            Features.mol_num_aliphatic_ring,
            Features.mol_num_aromatic_ring,
            Features.mol_mw,
            Features.mol_volume,
            Features.mol_num_atom
        ]

        self.atom += [
            Features.atom_is_h_acceptor,
            Features.atom_is_h_donor,
            Features.atom_crippen_log_p_contrib,
            Features.atom_crippen_molar_refractivity_contrib,
            Features.atom_symbol,
            Features.atom_hybridization,
            Features.atom_num_hs,
            Features.atom_num_valence,
            Features.atom_degree,
            Features.atom_is_aromatic,
            Features.atom_is_hetero,
            Features.atom_in_ring_size,
            Features.atom_radical_electrons,
            Features.atom_tpsa_contrib,
            Features.atom_labute_asa_contrib,
            Features.atom_gasteiger_charge
        ]
        
        self.bond += [
            Features.bond_type,
            Features.bond_is_conjugated,
            Features.bond_is_in_ring,
            Features.bond_is_rotatable,
            Features.bond_stereo
        ]

        self.sup += [
            Features.sup_SMILES,
            Features.sup_InChI,
            Features.sup_scaffold
        ]        
