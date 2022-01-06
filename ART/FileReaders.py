import pandas as pd
import itertools

from ART.funcs import split_list
from copy import deepcopy
from typing import Dict, List, Tuple, Union, Iterable, NamedTuple, Dict
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import AllChem


class MolRecord(NamedTuple):
    mol: Union[Mol, None]
    rt: float
    supplementary: Union[Dict, None]


class MolReader():
    def __init__(
            self, file_path, data: Union[List[MolRecord], None] = None,
            limit: Union[int, None]=None) -> None:
        self._file_path = file_path
        self._raw_data = self.truncate(x=self.read_file(), limit=limit)
        self._length = len(self._raw_data)

        if data is None:
            self._cache = [None] * self._length
        else:
            if len(data) == self._length:
                self._cache = data
            else:
                raise ValueError

        self._next_index = 0

    @property
    def file_path(self):
        return self._file_path

    @property
    def length(self) -> int:
        return len(self._raw_data)

    def __name__(self) -> str:
        return "MolReader"

    def __repr__(self) -> str:
        return f"{self.__name__}({self.length})"

    def __len__(self) -> int:
        return self.length

    @property
    def padding_obj(self):
        return MolRecord(
                mol="",
                rt=-1,
                supplementary={})

    def read_file(self) -> Iterable:
        raise NotImplementedError

    def process(self, obj) -> MolRecord:
        raise NotImplementedError

    def truncate(self, x: list, limit: Union[int, None]):
        if limit is None:
            return x
        else:
            limit = min(len(x), limit)
            return x[0:limit]

    def next(self) -> MolRecord:
        return self.__next__()

    def __getitem__(self, index) -> Union[MolRecord, List[MolRecord]]:
        if isinstance(index, slice):
            data = self._raw_data[index]
            cache = self._cache[index]
            out = [None] * len(data)
            for i, (d, c) in enumerate(zip(data, cache)):
                if c is None:
                    out[i] = self.process(d)
                else:
                    out[i] = c
            self._cache[index] = out
            return out
        else:
            if self._cache[index] is None:
                self._cache[index] = self.process(self._raw_data[index])
            return self._cache[index]

    def __next__(self) -> MolRecord:
        if self._next_index < self.length:
            index = self._next_index
            if self._cache[index] is None:
                self._cache[index] = self.process(self._raw_data[index])
            self._next_index += 1
            return self._cache[index]
        else:
            self._next_index = 0
            raise StopIteration

    def __iter__(self):
        return self


class SMRTSdfReader(MolReader):
    def __init__(
            self, 
            file_path: str, 
            data: Union[List[MolRecord], None] = None,
            sup_info: Dict = {}, 
            limit: Union[int, None] = None) -> None:
        self._num_error = 0
        self._sup_info = sup_info
        super().__init__(file_path=file_path, data=data, limit=limit)

    @property
    def __name__(self) -> str:
        return "SMRTSdfReader"

    def read_file(self) -> Iterable:
        mols = Chem.SDMolSupplier(self.file_path, removeHs=False)
        raw_data = [None] * len(mols)
        # RDLogger.DisableLog('rdApp.*')
        for i, mol in enumerate(mols):
            if (mol):
                raw_data[i] = {
                    "mol_block": Chem.MolToMolBlock(mol),
                    "mol_prop": {
                        "rt": mol.GetProp("RETENTION_TIME"),
                        "cid": mol.GetProp("PUBCHEM_COMPOUND_CID")}
                }
            else:
                self._num_error += 1
        # RDLogger.EnableLog('rdApp.info')
        return raw_data

    def process(self, obj) -> MolRecord:
        record = self.padding_obj
        try:
            if (obj is not None):
                mol = Chem.MolFromMolBlock(obj["mol_block"])
                if AllChem.EmbedMolecule(mol) == 0:
                    sup_info = self._sup_info
                    sup_info["cid"] = obj["mol_prop"]["cid"]
                    record = MolRecord(
                        mol=mol,
                        rt=float(obj["mol_prop"]["rt"]),
                        supplementary=sup_info
                    )
                else:
                    raise ValueError
            else:
                raise TypeError
        except ValueError:
            print("Mol could not be embeded.")
        except TypeError:
            print("SDF could not be transform to Mol.")
        except Exception as e:
            print(f"Unknown error: {e}")
        finally:
            return record


class CsvReader(MolReader):

    def __init__(
            self, file_path: str,
            data: Union[List[MolRecord], None] = None,
            notation_col: str = "inchi", rt_col: str = "rt",
            notation_method: str = "InChI",
            limit: Union[int, None] = None
            ) -> None:

        self._notation_method = notation_method
        self._notation_col = notation_col
        self._rt_col = rt_col
        super().__init__(file_path, data=data, limit=limit)

        if self._notation_method in ["SMILES", "smi", "smiles"]:
            self.molEncoder = Chem.MolFromSmiles
        elif self._notation_method  in ["InChI", "Inchi", "inchi"]:
            self.molEncoder = Chem.MolFromInchi
        else:
            ValueError("notation_method should be either SMILES or InChI")

    @property
    def notation_method(self):
        return self._notation_method

    @property
    def __name__(self) -> str:
        return "CsvReader"

    def read_file(self) -> Iterable:
        raw_data = pd.read_csv(self.file_path)
        raw_data_main = raw_data.loc[:, [self._notation_col, self._rt_col]]
        raw_data_sup = raw_data.drop([self._notation_col, self._rt_col], axis=1)
        raw_data_sup = raw_data_sup.to_dict(orient="records")
        raw_data_main["supplementary"] = raw_data_sup
        return raw_data_main.to_dict(orient="records")

    def process(self, obj) -> MolRecord:
        record = self.padding_obj
        try:
            mol = Chem.MolFromInchi(obj["inchi"])
            rt = obj["rt"]
            sup = obj["supplementary"]
            sup["rt_unit"] = "sec"

            if AllChem.EmbedMolecule(mol) == 0:
                record = MolRecord(
                    mol=mol,
                    rt=rt,
                    supplementary=sup
                )
            else:
                raise ValueError
        except ValueError:
            print("Mol could not be embeded.")
        except Exception as e:
            print(f"Unknown error: {e}")
        finally:
            return record


class ParallelMolReader():

    def __init__(self, n_jobs: int = 2, inplace: bool = False) -> None:
        self.n_jobs = n_jobs
        self.inplace = inplace
        self._mol_record = None
        self._mol_reader = None

    def __call__(self, mol_reader: MolReader, pool) -> MolReader:
        mol_record = pool.map(
            lambda x: mol_reader[x[0]:x[1]],
            self.batch_index(range(len(mol_reader)), n=self.n_jobs)
        )
        self._mol_record = list(itertools.chain.from_iterable(mol_record))
        self._mol_reader = mol_reader
        return self.file_reader

    def batch_index(self, x: List, n: int) -> List[Tuple]:
        return split_list(x=x, n=n, return_idxs=True)

    @property
    def file_reader(self):
        if self.inplace:
            mol_reader = self._mol_reader
        else:
            mol_reader = deepcopy(self._mol_reader)
        mol_reader._cache = self.mol_record
        return mol_reader

    @property
    def mol_record(self):
        return self._mol_record
