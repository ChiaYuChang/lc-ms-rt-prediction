# %%
from ART.funcs import calc_tilde_A
from collections import defaultdict
from typing import Callable, DefaultDict, Dict, List, Union, Dict
from numpy import Inf
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, Atom, Bond
from rdkit.Chem import AllChem, rdMolDescriptors, rdPartialCharges, Lipinski, Crippen
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

# Abstract type
# %%
class Feature():
    """
    An abstract type for different Feature

    Args:
    - name (str): the name of the feature
    - mapping (Union[None, Dict]): a dictionary for mapping categorical data
      to an integer. If None is given, the mapping process will be skipped.
    - func (Callable): a function that return a molecular discriptor
    - type (string): should be "atom", "bond", "mol" or "sup"
    """
    def __init__(
            self,
            name: str,
            mapping: Union[None, Dict],
            func: Callable,
            type: str) -> None:

        self._name = name
        self._mapping = mapping
        self._func = func
        self._type = type

        if isinstance(mapping, defaultdict):
            self._num_level = len(mapping) + 1
        elif isinstance(mapping, dict):
            self._num_level = len(mapping)
        else:
            self._num_level = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def mapping(self) -> Union[Dict, DefaultDict]:
        return self._mapping

    @property
    def func(self) -> Callable:
        return self._func

    @property
    def type(self) -> str:
        return self._type

    @property
    def num_level(self) -> Union[None, int]:
        return self._num_level

    def __repr__(self) -> str:
        out = f"Feature {{name={self.name}" +\
            f", func={self.func.__name__}\n" +\
            f", type={self.type}" +\
            f",num_level : {self.num_level}}}"
        return out


# %%
def gen_mapper(
        allowable_set: List, wo_defalut: bool = False, binary: bool = False
        ) -> Union[DefaultDict, Dict]:
    """Generate a dictionary for mapping categorical data to an integer

    Args:

    - allowable_set (List): Allowable keys for the dictionary
    - wo_defalut (bool): should 0 be the default when key in not in
      allowable_set
    - binary (bool): whether the tpye of the field is boolean
    """
    if binary:
        dct = {}
        for value, key in enumerate(allowable_set):
            dct[key] = int(value)
    else:
        if wo_defalut:
            dct = {}
            for value, key in enumerate(allowable_set):
                dct[key] = int(value)
        else:
            dct = defaultdict(lambda: 0)
            for value, key in enumerate(allowable_set):
                dct[key] = int(value) + 1
    return dct


def attr_truncator(
        func: Callable,
        lb: Union[int, float] = -Inf,
        ub: Union[int, float] = Inf
        ) -> Callable:
    """Generate a function that limit the range of the integer data

    When the given integer is less than lb, lb will be return.

    Args:
    - lb: the minimum value for the integer data
    - ub: the maximum value for the integer data
    """
    if lb > ub:
        ValueError("lb should be less than ub")

    def f(obj):
        x = func(obj)
        x = max(x, lb)
        x = min(x, ub)
        return x
    f.__name__ = func.__name__
    return f

# %%
# Mol Feature
mol_num_h_acceptor = Feature(
    name="num_h_acceptor",
    func=lambda x: attr_truncator(
        func=rdMolDescriptors.CalcNumHBA,
        lb=0,
        ub=5
    )(x),
    type="discrete",
    mapping=gen_mapper(
        allowable_set=list(range(0, 11)),
        wo_defalut=True
    )
)

mol_num_h_donor = Feature(
    name="num_h_donor",
    func=lambda x: attr_truncator(
        func=rdMolDescriptors.CalcNumHBD,
        lb=0,
        ub=4
    )(x),
    type="discrete",
    mapping=gen_mapper(
        allowable_set=list(range(0, 5)),
        wo_defalut=True
    )
)

mol_num_ring = Feature(
    name="num_ring",
    func=lambda x: attr_truncator(
        func=rdMolDescriptors.CalcNumRings,
        lb=0,
        ub=6
    )(x),
    type="discrete",
    mapping=gen_mapper(
        allowable_set=list(range(0, 7)),
        wo_defalut=True
    )
)

mol_num_aromatic_ring = Feature(
    name="num_aromatic_ring",
    func=lambda x: attr_truncator(
        func=rdMolDescriptors.CalcNumAromaticRings,
        lb=0,
        ub=5
    )(x),
    type="discrete",
    mapping=gen_mapper(
        allowable_set=list(range(0, 6)),
        wo_defalut=True
    )
)

mol_num_aliphatic_ring = Feature(
    name="num_aliphatic_ring",
    func=lambda x: attr_truncator(
        func=rdMolDescriptors.CalcNumAliphaticRings,
        lb=0,
        ub=4
    )(x),
    type="discrete",
    mapping=gen_mapper(
        allowable_set=list(range(0, 5)),
        wo_defalut=True
    )
)

mol_mw = Feature(
    name="mw",
    func=lambda x: rdMolDescriptors.CalcExactMolWt(x),
    type="continuous",
    mapping=None
)

mol_volume = Feature(
    name="volume",
    func=lambda x: AllChem.ComputeMolVolume(x),
    type="continuous",
    mapping=None
)

mol_num_atom = Feature(
    name="num_nodes",
    func=lambda x: Mol.GetNumAtoms(x),
    type="discrete",
    mapping=None
)

# Atom Feature
# Symbol
atom_symbol = Feature(
    name="symbol",
    func=lambda x: Atom.GetSymbol(x),
    type="categorical",
    mapping=gen_mapper(
        allowable_set=[
            "C", "N", "O", "S", "F",
            "Cl", "Br", "I", "P", "Si"
        ]
    )
)

# Hybridization
atom_hybridization = Feature(
    name="hybridization",
    func=lambda x: Atom.GetHybridization(x),
    type="categorical",
    mapping=gen_mapper(
        allowable_set=[
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3
        ]
    )
)

# Number of hydrogen connecto to an atom
atom_num_hs = Feature(
    name="num_hs",
    func=lambda x: attr_truncator(
        func=Atom.GetTotalNumHs, lb=0, ub=4)(x),
    type="discrete",
    mapping=gen_mapper(
        allowable_set=list(range(0, 5)),
        wo_defalut=True
    )
)

# Total valence (implicit and explicit) for an atom
atom_num_valence = Feature(
    name="num_valence",
    func=lambda x: attr_truncator(
        func=Atom.GetTotalValence, lb=0, ub=6)(x),
    type="discrete",
    mapping=gen_mapper(
        allowable_set=list(range(0, 7)),
        wo_defalut=True
    )
)

# Number of bonded neighbors in the graph for a atom
atom_degree = Feature(
    name="degree",
    func=lambda x: attr_truncator(
        func=Atom.GetDegree, lb=0, ub=5)(x),
    type="discrete",
    mapping=gen_mapper(
        allowable_set=list(range(0, 6)),
        wo_defalut=True
    )
)

atom_is_aromatic = Feature(
    name="is_aromatic",
    func=lambda x: Atom.GetIsAromatic(x),
    type="boolean",
    mapping=gen_mapper(
        allowable_set=[False, True],
        binary=True
    )
)


def is_hetero(atom: Chem.Atom) -> bool:
    mol = atom.GetOwningMol()
    return atom.GetIdx() in [i[0] for i in Lipinski._Heteroatoms(mol)]


atom_is_hetero = Feature(
    name="is_hetero",
    func=is_hetero,
    type="boolean",
    mapping=gen_mapper(
        allowable_set=[False, True],
        binary=True
    )
)


def is_h_donor(atom: Chem.Atom) -> bool:
    mol = atom.GetOwningMol()
    return atom.GetIdx() in [i[0] for i in Lipinski._HDonors(mol)]


atom_is_h_donor = Feature(
    name="is_h_donor",
    func=is_h_donor,
    type="boolean",
    mapping=gen_mapper(
        allowable_set=[False, True],
        binary=True
    )
)


def is_h_acceptor(atom: Chem.Atom) -> bool:
    mol = atom.GetOwningMol()
    return atom.GetIdx() in [i[0] for i in Lipinski._HAcceptors(mol)]


atom_is_h_acceptor = Feature(
    name="is_h_acceptor",
    func=is_h_acceptor,
    type="boolean",
    mapping=gen_mapper(
        allowable_set=[False, True],
        binary=True
    )
)


def in_ring_size(atom: Atom):
    for ring_size in [10, 9, 8, 7, 6, 5, 4, 3, 0]:
        if Atom.IsInRingSize(atom, ring_size):
            break
    return ring_size


atom_in_ring_size = Feature(
    name="in_ring_size",
    func=lambda x: attr_truncator(
        func=in_ring_size, lb=2, ub=10)(x),
    type="discrete",
    mapping=gen_mapper(
        allowable_set=list(range(2, 11)),
        wo_defalut=True
    )
)

atom_radical_electrons = Feature(
    name="radical_electrons",
    func=lambda x: attr_truncator(
        Atom.GetNumRadicalElectrons, lb=0, ub=2
    )(x),
    type="discrete",
    mapping=gen_mapper(
        allowable_set=[0, 1, 2],
        wo_defalut=True
    )
)


# Partition coefficient (log P)
def crippen_log_p_contrib(atom: Atom) -> float:
    mol = atom.GetOwningMol()
    return Crippen._GetAtomContribs(mol)[atom.GetIdx()][0]


atom_crippen_log_p_contrib = Feature(
    name="crippen_log_p_contrib",
    func=crippen_log_p_contrib,
    type="continuous",
    mapping=None
)


# a measure of the steric effect
def crippen_molar_refractivity_contrib(atom: Chem.Atom) -> float:
    mol = atom.GetOwningMol()
    return Crippen._GetAtomContribs(mol)[atom.GetIdx()][1]


atom_crippen_molar_refractivity_contrib = Feature(
    name="crippen_molar_refractivity_contrib",
    func=crippen_molar_refractivity_contrib,
    type="continuous",
    mapping=None
)


# a 2D-QSAR descriptor that represents a relationship between ligands
# and specific targets (Prasanna and Doerksen, 2008)
def tpsa_contrib(atom: Chem.Atom) -> float:
    mol = atom.GetOwningMol()
    return rdMolDescriptors._CalcTPSAContribs(mol)[atom.GetIdx()]


atom_tpsa_contrib = Feature(
    name="tpsa_contrib",
    func=tpsa_contrib,
    type="continuous",
    mapping=None
)


# The accessible surface area (ASA) or solvent-accessible surface
# area (SASA) is the surface area of a biomolecule that is accessible
# to a solvent.
def labute_asa_contrib(atom: Chem.Atom) -> float:
    mol = atom.GetOwningMol()
    return rdMolDescriptors._CalcLabuteASAContribs(mol)[0][atom.GetIdx()]


atom_labute_asa_contrib = Feature(
    name="labute_asa_contrib",
    func=labute_asa_contrib,
    type="continuous",
    mapping=None
)


def gasteiger_charge(atom: Chem.Atom) -> float:
    mol = atom.GetOwningMol()
    rdPartialCharges.ComputeGasteigerCharges(mol)
    return atom.GetDoubleProp('_GasteigerCharge')


atom_gasteiger_charge = Feature(
    name="gasteiger_charge",
    func=gasteiger_charge,
    type="continuous",
    mapping=None
)

# Bond Feature
# bond type
bond_type = Feature(
    name="bond_type",
    func=lambda x: Bond.GetBondType(x),
    type="categorical",
    mapping=gen_mapper(
        allowable_set=[
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]
    )
)

# whether a bond is in ring(s)
bond_is_in_ring = Feature(
    name="in_ring",
    func=lambda x: Bond.IsInRing(x),
    type="boolean",
    mapping=gen_mapper(
        allowable_set=[False, True],
        binary=True
    )
)

# whether a bond is conjugated
bond_is_conjugated = Feature(
    name="is_conjugated",
    func=lambda x: Bond.GetIsConjugated(x),
    type="boolean",
    mapping=gen_mapper(
        allowable_set=[False, True],
        binary=True
    )
)


# whether a bond is rotatable
def is_rotatable(bond: Chem.Bond) -> List[float]:
    mol = bond.GetOwningMol()
    atom_indices = tuple(
        sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))
    return atom_indices in Lipinski._RotatableBonds(mol)


bond_is_rotatable = Feature(
    name="is_rotatable",
    func=is_rotatable,
    type="boolean",
    mapping=gen_mapper(
        allowable_set=[False, True],
        binary=True
    )
)

# stereo
bond_stereo = Feature(
    name="stereo",
    func=lambda x: Bond.GetStereo(x),
    type="categorical",
    mapping=gen_mapper(
        allowable_set=[
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOANY,
        ]
    )
)

# Supplementary Feature
sup_SMILES = Feature(
    name="SMILES",
    func=lambda x: Chem.MolToSmiles(x),
    type="id",
    mapping=None
)


def gen_scaffold(mol: Mol):
    return MurckoScaffoldSmiles(mol=mol, includeChirality=False)


sup_scaffold = Feature(
    name="scaffold",
    func=gen_scaffold,
    type="scaffold",
    mapping=None
)

sup_InChI = Feature(
    name="InChI",
    func=lambda x: Chem.MolToInchi(x),
    type="id",
    mapping=None
)

# Graph features
graph_tilde_A = Feature(
    name="tilde_A",
    func=calc_tilde_A,
    type="tilde_A",
    mapping=None
)
