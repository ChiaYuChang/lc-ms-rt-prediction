# %%
import pandas as pd
import numpy as np
import copy
import itertools
import pymongo
import json

from bson.objectid import ObjectId
from collections import defaultdict
from datetime import datetime
from joblib import Parallel, delayed
from pymongo import MongoClient
from pymongo import collation
from pymongo.message import query
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Crippen
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from tqdm import tqdm
from typing import Dict, List, Union
from pymongo.errors import ConnectionFailure


def test_mng_connection(mng_client):
    try:
        mng_client.admin.command('ismaster')
        return True
    except ConnectionFailure:
        print("Server not available")
        return False


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


def inchi2mol(inchi):
    try:
        return(Chem.MolFromInchi(inchi))
    except:
        return(None)


# %%
def gen_mol_data(
        mol: Mol,
        mol_formula: str,
        mol_volume: float,
        rt_data_id: ObjectId,
        mol_wt: float,
        tautomer: List,
        is_tautomers: bool
        ) -> Dict:
    # Number of atom in the mol
    mol_n_node = mol.GetNumAtoms()
    # Number of bond in the mol
    mol_n_edge = mol.GetNumBonds()
    # SMILES
    mol_SMILES = Chem.MolToSmiles(mol)
    # InChI
    mol_InChI = Chem.MolToInchi(mol)
    # InChIKey
    mol_InChIKey = Chem.MolToInchiKey(mol)
    # Number of hydrogen bond acceptor
    mol_n_hba = rdMolDescriptors.CalcNumHBA(mol)
    # Number of hydrogen bond donor
    mol_n_hbd = rdMolDescriptors.CalcNumHBD(mol)
    # Number of ring
    mol_n_ring = rdMolDescriptors.CalcNumRings(mol)
    # Number of aromatic ring
    mol_n_aromatic_ring = rdMolDescriptors.CalcNumAromaticRings(mol)
    # Number of aromatic ring
    mol_n_aliphatic_ring = rdMolDescriptors.CalcNumAliphaticRings(mol)
    mol_molLogP = Crippen.MolLogP(mol)
    mol_scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=False)

    mol_attrs = {
        "SMILES": mol_SMILES,
        "InChI": mol_InChI,
        "InChIKey": mol_InChIKey,
        "formula": mol_formula,
        "rt_data_id": rt_data_id,
        "wt": mol_wt,
        "volume": mol_volume,
        "n_hba": mol_n_hba,
        "n_hbd": mol_n_hbd,
        "n_ring": mol_n_ring,
        "n_aromatic_ring": mol_n_aromatic_ring,
        "n_aliphatic_ring": mol_n_aliphatic_ring,
        "n_edge": mol_n_edge,
        "n_node": mol_n_node,
        "mLogP": mol_molLogP,
        "scaffold": mol_scaffold,
        "is_tautomers": is_tautomers,
        "tautomer": tautomer,
    }
    return(mol_attrs)


# %%
def dot_N(x):
    if isinstance(x, pd.DataFrame):
        return list(range(x.shape[1]))
    else:
        return list(range(len(x)))


def gen_atom_attrs(mol: Mol, embedding_flag, hybridization_dict: Dict):
    atom = pd.DataFrame({
        "index": range(0, mol.GetNumAtoms()),
        "atom": [atom for atom in mol.GetAtoms()],
    })

    atom.set_index("index", inplace=True)
    atom["symbol"] = [atom.GetSymbol() for atom in atom.atom]
    atom["hybridization"] = [
        hybridization_dict[atom.GetHybridization()] for atom in atom.atom
    ]
    atom["degree"] = [atom.GetDegree() for atom in atom.atom]
    atom["n_hs"] = [atom.GetTotalNumHs() for atom in atom.atom]
    atom["formal_charge"] = [atom.GetFormalCharge() for atom in atom.atom]
    atom["aromaticity"] = [
        1 if atom.GetIsAromatic() else 0 for atom in atom.atom
    ]
    atom["in_ring"] = [1 if atom.IsInRing() else 0 for atom in atom.atom]
    atom.drop(labels=["atom"], axis=1, inplace=True)
    atom.reset_index(drop=False, inplace=True)
    atomIdx2Sym = {}
    for i, s in zip(atom.index, atom.symbol):
        atomIdx2Sym[i] = s
    atom = atom.to_dict(orient="records")

    if embedding_flag == 0:
        dist = pd.DataFrame(Chem.Get3DDistanceMatrix(mol))
        dist["from_atom_index"] = range(mol.GetNumAtoms())
        dist = dist.melt(
            "from_atom_index", var_name="to_atom_index", value_name="distance"
        )
        dist = dist[dist["from_atom_index"] != dist["to_atom_index"]]
        dist.reset_index(drop=True, inplace=True)
        dist["from_atom_symbol"] = [
            atomIdx2Sym[i] for i in dist.from_atom_index
        ]
        dist["to_atom_symbol"] = [atomIdx2Sym[i] for i in dist.to_atom_index]
        dist.sort_values(["from_atom_index", "to_atom_index"], inplace=True)
        dist = dist.reindex(
            columns=["from_atom_index", "from_atom_symbol",
                     "to_atom_index", "to_atom_symbol", "distance"]
        )
        dist.reset_index(inplace=True)
        dist.sort_values(["from_atom_index", "distance"], inplace=True)
        dist["k"] = dist.groupby("from_atom_index").index.transform(dot_N)
        dist.drop(columns="index", inplace=True)
        dist = dist.to_dict(orient="records")
        atom_attrs = {
            "atom_attrs": atom,
            "embedding_flag": embedding_flag,
            "distance": dist
        }
    else:
        atom_attrs = {
            "atom_attrs": atom,
            "embedding_flag": embedding_flag,
            "distance": None
        }
    return(atom_attrs)


# %%
def gen_bond_attrs(mol, bondtype_dict, stereos_dict):
    edge = pd.DataFrame({
                "index": range(0, mol.GetNumBonds()),
                "bond": [bond for bond in mol.GetBonds()],
            })
    # print("Extracting bond features")
    edge["bondtype"] = [
        bondtype_dict[bond.GetBondType()] for bond in edge.bond
    ]
    edge["conjugation"] = [bond.GetIsConjugated() for bond in edge.bond]
    edge["in_ring"] = [bond.IsInRing() for bond in edge.bond]
    edge["stereos"] = [stereos_dict[bond.GetStereo()] for bond in edge.bond]
    edge["aromatic"] = [bond.GetIsAromatic() for bond in edge.bond]

    bond_begin_atom_idx = [bond.GetBeginAtomIdx() for bond in edge.bond]
    bond_end_atom_idx = [bond.GetEndAtomIdx() for bond in edge.bond]

    edge_index = pd.DataFrame({
        "begin": bond_begin_atom_idx + bond_end_atom_idx,
        "end": bond_end_atom_idx + bond_begin_atom_idx
    })
    edge_index = edge_index.to_dict(orient="records")
    edge.drop(labels=["bond"], axis=1, inplace=True)
    edge = edge.append(edge)
    edge = edge.to_dict(orient="records")
    return({"edge_attrs": edge, "edge_index": edge_index})


# %%
def gen_attrs(
        inchi: str,
        obj_ids,
        hybridization_dict: Dict,
        bondtype_dict: Dict,
        stereos_dict: Dict,
        max_tt: int
        ) -> List:
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(max_tt)

    mol = Chem.MolFromInchi(inchi)
    rt_data_id = np.unique(obj_ids).tolist()
    flg = Chem.AllChem.EmbedMolecule(mol)
    mol_formula = rdMolDescriptors.CalcMolFormula(mol)
    mol_wt = rdMolDescriptors.CalcExactMolWt(mol)
    mol_volume = None
    mol_attrs_list = []
    if flg == 0:
        mol_volume = np.round(
            Chem.AllChem.ComputeMolVolume(mol),
            decimals=4
        )
        tautomers = [mol for mol in enumerator.Enumerate(mol)]
    else:
        tautomers = [mol]

    mol_InChI = [Chem.MolToInchi(mol) for mol in tautomers]
    # mol_SMILE = np.unique(mol_SMILE).tolist()
    for i, mol in enumerate(tautomers):
        if i == 0:
            is_tautomers = False
        else:
            is_tautomers = True
        mol_attrs = {}
        mol_attrs["mol_attrs"] = gen_mol_data(
            mol=mol,
            mol_formula=mol_formula,
            mol_volume=mol_volume,
            rt_data_id=rt_data_id,
            mol_wt=mol_wt,
            tautomer=mol_InChI,
            is_tautomers=is_tautomers
        )

        mol_attrs["atom_attrs"] = gen_atom_attrs(
            mol=mol,
            embedding_flag=flg,
            hybridization_dict=hybridization_dict
        )

        mol_attrs["edge_attrs"] = gen_bond_attrs(
            mol=mol,
            bondtype_dict=bondtype_dict,
            stereos_dict=stereos_dict
        )
        mol_attrs_list.append(mol_attrs)
    return(mol_attrs_list)


# %%
def gen_attrs_batch(
        inchi: Union[list, pd.Series],
        obj_ids: Union[list, pd.Series],
        hybridization_dict: dict,
        bondtype_dict: dict,
        stereos_dict: dict,
        max_tt: int,
        conn_str: Union[str, None],
        database: Union[str, None],
        collection: Union[str, None],
        dry_run: bool = False
        ):
    result_vec = []

    with MongoClient(conn_str, connectTimeoutMS=None) as mng_client:
        mng_db = mng_client[database]
        mng_col = mng_db[collection]

        for i, i_id in zip(inchi, obj_ids):
            doc_list = gen_attrs(
                inchi=i,
                obj_ids=i_id,
                hybridization_dict=hybridization_dict,
                bondtype_dict=bondtype_dict,
                stereos_dict=stereos_dict,
                max_tt=max_tt
            )

            if (dry_run is None):
                result_vec.append({
                    "inich": i,
                    "rt_obj_id": i_id,
                    "doc": doc_list
                })
            else:
                insert_result = mng_col.insert_many(doc_list)
                error_counter = 0
                while (insert_result.acknowledged is False) and (error_counter < 10):
                    print(f"Insert error, try again... ({error_counter:02d}/10)")
                    insert_result = mng_col.insert_many(doc_list)

                result_vec.append({
                    "inich": i,
                    "rt_obj_id": i_id,
                    "mol_attr_obj_id": insert_result.inserted_ids
                })
    return(result_vec)


# %%
def insert_rt_data(
        df: pd.DataFrame,
        desc: str,
        conn_str: str,
        database: str,
        collection: str,
        default: defaultdict = defaultdict(lambda: None)
        ) -> pd.DataFrame:

    with MongoClient(conn_str) as mng_client:
        mng_db = mng_client[database]
        mng_col = mng_db[collection]
        num_row = df.shape[0]
        doc_list = [None] * num_row
        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=num_row, desc=desc)):
            cid = row.cid
            if cid == "":
                cid = None

            doc = {}
            for colname in ['rt_unit', 'mol_name', 'system',
                            'username', 'upload_date']:
                doc[colname] = row[colname] if colname in df.columns else default[colname]

            doc["upload_date"] = datetime.strptime(
                doc["upload_date"] + "T00:00:00+00:00", "%Y-%m-%dT%H:%M:%S%z"
            )
            mol = row.mol
            doc_list[i] = {**{
                "SMILE": Chem.MolToSmiles(mol),
                "InChI": Chem.MolToInchi(mol),
                "InChIKey": Chem.MolToInchiKey(mol),
                "PubChemCid": cid,
                "rt": np.round(row.rt, 2),
                "database": database
            }, **doc}
        insert_result = mng_col.insert_many(doc_list)
        error_counter = 0
        while (insert_result.acknowledged is False) and (error_counter < 10):
            print(f"Insert error, try again... ({error_counter:02d}/10)")
            insert_result = mng_col.insert_many(doc_list)

    result = pd.DataFrame({
        "index": list(df.index),
        "obj_id": insert_result.inserted_ids
    })
    result.set_index(keys="index", inplace=True)
    return(result)


# %%
def batch_insert_rt_data(
        df: pd.DataFrame,
        desc: str,
        conn_str: str,
        database: str,
        collection: str,
        default: defaultdict = defaultdict(lambda: None),
        n_job: int = 1,
        max_batch_size: int = 1000
        ) -> pd.DataFrame:
    if (df.shape[0] // n_job + 1 > max_batch_size):
        df_list = np.array_split(df, df.shape[0] // 1000 + 1)
    else:
        df_list = np.array_split(df, n_job)

    inserted_ids_list = Parallel(n_jobs=n_job)(delayed(insert_rt_data)(
        df=df,
        desc=desc,
        conn_str=conn_str,
        database=database,
        collection=collection,
        default=default
    ) for df in df_list)
    return(pd.concat(inserted_ids_list))


# %%
# Parallel setting
N_JOB = 15
MAX_BATCH_SIZE = 10000

# Read login info
print("Setup connection")
input("Press Enter to continue...")

with open("/home/cychang/.mongo_login_info.json") as f:
    login_info = json.load(f)

REPLICA_NAME = login_info["replicaName"]
MONGO_USERNAME = login_info["username"]
MONGO_PASSWORD = login_info["password"]
MONGO_HOSTS = ",".join(
    [host["host"] + ":" + str(host["port"])
        for host in login_info["hosts"]]
)
MONGO_AUTH_DB = login_info["authenticationDatabase"]
MONGO_READ_PREFERENCE = "primary"
MONGO_CONN_STR = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOSTS}/?authSource={MONGO_AUTH_DB}&replicaSet={REPLICA_NAME}&readPreference={MONGO_READ_PREFERENCE}"
MONGO_DB = "mols"
RT_COLLECTION_NAME = "RT_data_new"
MOLATTRS_COLLECTION_NAME = "MolAttrs_new"
DROP_EXIST_COLLECTION = True
LIMIT = 100

with MongoClient(MONGO_CONN_STR) as mng_client:
    mng_db = mng_client[MONGO_DB]
    # Create collection if RT_COLLECTION does not exist
    if (RT_COLLECTION_NAME in set(mng_db.list_collection_names())) and DROP_EXIST_COLLECTION:
        mng_db.drop_collection(name_or_collection=RT_COLLECTION_NAME)
    mng_db.create_collection(name=RT_COLLECTION_NAME)
    # Create collection if MOLATTRS_COLLECTION_NAME does not exist
    if MOLATTRS_COLLECTION_NAME in set(mng_db.list_collection_names()) and DROP_EXIST_COLLECTION:
        mng_db.drop_collection(name_or_collection=MOLATTRS_COLLECTION_NAME)
    mng_db.create_collection(name=MOLATTRS_COLLECTION_NAME)

# %%
bondtype_dict = defaultdict(lambda: "others")
bondtype_dict[Chem.rdchem.BondType.SINGLE] = "single"
bondtype_dict[Chem.rdchem.BondType.DOUBLE] = "double"
bondtype_dict[Chem.rdchem.BondType.TRIPLE] = "triple"
bondtype_dict[Chem.rdchem.BondType.AROMATIC] = "aromatic"

stereos_dict = defaultdict(lambda: "others")
stereos_dict[Chem.rdchem.BondStereo.STEREOE] = "stereoe"
stereos_dict[Chem.rdchem.BondStereo.STEREONONE] = "stereonone"
stereos_dict[Chem.rdchem.BondStereo.STEREOZ] = "stereoz"

hybridization_dict = defaultdict(lambda: "others")
hybridization_dict[Chem.rdchem.HybridizationType.SP] = "sp"
hybridization_dict[Chem.rdchem.HybridizationType.SP2] = "sp2"
hybridization_dict[Chem.rdchem.HybridizationType.SP3] = "sp3"

# %%
# Read PredRet data
print("Add Predret RT")
input("Press Enter to continue...")

predret = pd.read_csv("./Data/PredRet.csv")
predret['rt'] = predret.rt * 60
predret = predret[predret["inchi"] != ""]
predret_w_cid = predret[predret.cid.notna()].copy()
predret_wo_cid = predret[predret.cid.isna()].copy()
predret_w_cid["cid"] = predret_w_cid.cid.astype(int).astype(str)
predret_wo_cid["cid"] = predret_wo_cid.cid.astype("str")
predret = predret_w_cid.append(predret_wo_cid)
predret.drop(labels="inS1", axis=1, inplace=True)
predret["mol"] = [inchi2mol(inchi) for inchi in predret.inchi]
predret["mol_name"] = predret.name.astype(str)
predret.rename(columns={"date": "upload_date"}, inplace=True)
if LIMIT is not None:
    predret = predret.iloc[0:LIMIT, :]
default = defaultdict(lambda: None)
default["rt_unit"] = "sec"
predret = predret[[isinstance(mol, Chem.rdchem.Mol) for mol in predret.mol]]
predret_inserted_ids = batch_insert_rt_data(
    df=predret,
    desc="Inserting rt data from Predret Database",
    conn_str=MONGO_CONN_STR,
    database=MONGO_DB,
    collection=RT_COLLECTION_NAME,
    n_job=3,
    max_batch_size=MAX_BATCH_SIZE,
    default=default
)
predret = predret.loc[:, ["inchi", "mol"]]
predret = predret.join(predret_inserted_ids)

# %%
print("Add SMRT RT")
input("Press Enter to continue...")

SDF_spplr = Chem.SDMolSupplier("./Data/SMRT_dataset.sdf")
SMRT = pd.DataFrame({
    "mol": [mol for mol in SDF_spplr if isinstance(mol, Chem.rdchem.Mol)]
})
SMRT["cid"] = [mol.GetProp("PUBCHEM_COMPOUND_CID") for mol in SMRT.mol]
SMRT["inchi"] = [Chem.MolToInchi(mol) for mol in SMRT.mol]
SMRT["rt"] = [mol.GetProp("RETENTION_TIME") for mol in SMRT.mol]
SMRT["rt"] = SMRT.rt.astype(float)
if LIMIT is not None:
    SMRT = SMRT.iloc[0:LIMIT, :]

default = defaultdict(lambda: None)
default["rt_unit"] = "sec"
default["upload_date"] = "2019-12-20"
default["name"] = "SMRT"
default['system'] = "Agilent 1100/1200 series liquid chromatography (LC) system"
default['username'] = "Xavier Domingo"

SMRT_inserted_ids = batch_insert_rt_data(
    df=SMRT,
    desc="Inserting rt data from Predret Database",
    conn_str=MONGO_CONN_STR,
    database=MONGO_DB,
    collection=RT_COLLECTION_NAME,
    n_job=10,
    max_batch_size=MAX_BATCH_SIZE,
    default=default
)

SMRT = SMRT.loc[:, ["inchi", "mol"]]
SMRT = SMRT.join(SMRT_inserted_ids)

# %%
print("Add Self Lab RT")
input("Press Enter to continue...")

self_lab = pd.read_csv("./Data/self_lab_w_smi.csv")
self_lab = self_lab.rename(columns={"pubchem": "cid"},)
self_lab["mol"] = [Chem.MolFromSmiles(smi) for smi in self_lab.SMILES]
self_lab["inchi"] = [Chem.MolToInchi(mol) for mol in self_lab.mol]
if LIMIT is not None:
    self_lab = self_lab.iloc[0:LIMIT, :]

# %%
default = defaultdict(lambda: None)
default["rt_unit"] = "sec"
default["upload_date"] = "2010-01-01"
default["name"] = "SELF"
default['system'] = "liquid chromatography system"
default['username'] = "SELF"

# %%
self_lab_inserted_ids = batch_insert_rt_data(
    df=self_lab,
    desc="Inserting rt data from Self Lab",
    conn_str=MONGO_CONN_STR,
    database=MONGO_DB,
    collection=RT_COLLECTION_NAME,
    n_job=2,
    max_batch_size=MAX_BATCH_SIZE,
    default=default
)
self_lab = self_lab.loc[:, ["inchi", "mol"]]
self_lab = self_lab.join(self_lab_inserted_ids)

# %%
print("Add Mol attributes")
input("Press Enter to continue...")

all_mol = pd.concat([predret, SMRT, self_lab])
all_mol = all_mol.groupby("inchi")[["obj_id"]]\
    .agg(lambda x: np.unique(list(itertools.chain(x))).tolist())
all_mol = all_mol.reset_index(drop=False)

batch_size = all_mol.shape[0] // N_JOB
if (batch_size > MAX_BATCH_SIZE):
    batch_size = MAX_BATCH_SIZE
batch_start = np.array(range(15)) * batch_size
batch_start = batch_start.astype(int)
batch_end = np.append(copy.deepcopy(batch_start[1:]), all_mol.shape[0])
print(f"batch size: {batch_size}, n jobs: {N_JOB}")

# %%
all_mol_attrs_list = Parallel(n_jobs=N_JOB)(delayed(gen_attrs_batch)(
    inchi=all_mol.loc[strt:end, "inchi"],
    obj_ids=all_mol.loc[strt:end, "obj_id"],
    hybridization_dict=hybridization_dict,
    bondtype_dict=bondtype_dict,
    stereos_dict=stereos_dict,
    max_tt=10,
    conn_str=MONGO_CONN_STR,
    database=MONGO_DB,
    collection=MOLATTRS_COLLECTION_NAME,
    dry_run=False
) for (strt, end) in zip(batch_start, batch_end))


# %%
def add_tt_obj_id(
        mol_attrs_ids: List,
        conn_str: str,
        database: str,
        collection: str
        ) -> None:
    with MongoClient(conn_str) as mng_client:
        mng_db = mng_client[database]
        mng_col = mng_db[collection]
        for doc in tqdm(mol_attrs_ids):
            for rt_obj_id in doc["rt_obj_id"]:
                if mng_col.count_documents({"_id": rt_obj_id}) == 1:
                    query = {"_id": rt_obj_id}
                    new_value = {
                        "$set": {
                            "mol_attrs_id": doc["mol_attr_obj_id"]
                        }
                    }
                    mng_col.update_one(filter=query, update=new_value)
                else:
                    ValueError("ID error")
    return(None)

print("Add Mol id to RT data")
input("Press Enter to continue...")

Parallel(n_jobs=N_JOB)(delayed(add_tt_obj_id)(
    mol_attrs_ids=mol_attrs_ids,
    conn_str=MONGO_CONN_STR,
    database=MONGO_DB,
    collection=RT_COLLECTION_NAME,
) for mol_attrs_ids in all_mol_attrs_list)

print(f"Number of records {all_mol.shape[0]}\n")
# %%
