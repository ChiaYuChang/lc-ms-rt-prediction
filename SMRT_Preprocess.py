import pandas as pd
import numpy as np
import torch
import pickle

from ART.DataSet import SMRT
from ART.DataSplitter import RandomSplitter
# from ART.DataSplitter import ScaffoldSplitter
from ART.Featurizer.FeatureSet import DefaultFeatureSet
from ART.Featurizer.Featurizer import Featurizer, ParallelFeaturizer
from ART.FileReaders import ParallelMolReader
from ART.FileReaders import SMRTSdfReader
from ART.DataTransformer.DataTransformer import DataTransformer
from ART.DataTransformer.Transforms import gen_normalized_adj_matrix
from ART.DataTransformer.Transforms import gen_knn_graph, gen_knn_distance
from ART.DataTransformer.Transforms import gen_radius_graph, gen_radius_distance
from ART.funcs import check_has_processed, data_to_doc, doc_to_data
from ART.DataTransformer.Transforms import gen_mw_mask, gen_mw_ppm

from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool
from rdkit import RDLogger

def pre_filter(data, thr: float = 300.0) -> bool:
    if data.y.dim() == 0:
        y = data.y
    elif data.y.dim() == 1:
        y = data.y[0]
    else:
        raise ValueError("y should be a scalar or a 1 element 1D tensor.")
    
    if y > thr: # 5 mins 
        return True
    else:
        return False

if __name__ == '__main__':
    root = "./Data/SMRT"
    raw_file_names = "SMRT_dataset.sdf"
    prefix = "    - "
    
    print("(1/7) Checking files")
    if not(check_has_processed(
            root=root, raw_file_names=raw_file_names,
            processed_file_names=["pre_filter.pt",  "pre_transform.pt",  "test.pt", 
                                  "train.pt",  "valid.pt",  "smrt_mw.pt"],
            indent= "    "
        )):
        print(prefix + f"Calculating descriptors")
        n_jobs = min(cpu_count() // 4 * 3, 50)
        print(prefix + f"Using {n_jobs} cores for preprocessing")

        sup_info = {
            "system": "Agilent 1100/1200 series liquid chromatography (LC) system",
            "username": "Xavier Domingo",
            "upload_date": "2019-12-20",
            "rt_unit": "sec"
        }

        RDLogger.DisableLog('rdApp.*')
    
        print("(2/7) Setting up Reader and Featurizer")
        mol_reader = SMRTSdfReader(
            file_path="/".join((root, "raw", raw_file_names)),
            max_tautomer=10,
            sup_info = sup_info
        )

        parallel_mol_reader = ParallelMolReader(
            n_jobs=n_jobs,
            inplace=False
        )

        featurizer = Featurizer(
            feature_set=DefaultFeatureSet(),
            include_coordinates=True,
            use_np = True
        )

        parallel_featureizer = ParallelFeaturizer(
            featurizer=featurizer,
            n_jobs=n_jobs,
            rm_None=True
        )

        pool = ProcessingPool(nodes=n_jobs)

        print("(3/7) Reading File")
        smrt = parallel_mol_reader(mol_reader=mol_reader, pool=pool)

        print("(4/7) Featurizing")
        smrt = parallel_featureizer(file_reader=smrt, pool=pool)

        smrt_df = pd.DataFrame.from_records([{"mw": d.sup["mw"], "rt": d.sup["rt"]} for d in smrt])
        smrt_df.sort_values(["mw", "rt"], inplace=True)
        smrt_df.reset_index(inplace=True)
        smrt_mw = np.array(smrt_df["mw"])
        smrt_y_category = np.array(smrt_df["index"])
        smrt_n = smrt_df.shape[0]
        for i, y_cat in enumerate(smrt_y_category):
            smrt[i]["y_cat"] = np.array([y_cat, smrt_n], dtype=np.compat.long)
            smrt[i]["sup"]["one_hot"] = y_cat
        
        with open("/".join([root, "processed", "smrt_mw.pt"]), "wb") as f:
            pickle.dump(torch.tensor(smrt_mw, dtype=torch.float32), f)
        
        print("(5/7) Converting np.array to torch.tensor")
        smrt_doc = pool.map(data_to_doc, smrt)
        smrt = pool.map(lambda d: doc_to_data(d,  w_sup_field=True, zero_thr=1.0), smrt_doc)

        print("(6/7) Setting pre-transform function")
        with open("/".join([root, "processed", "smrt_mw.pt"]), "rb") as f:
            smrt_mw = pickle.load(f)

        gen_mw_ppm.args["mw_list"] =  smrt_mw 
        gen_mw_mask.args["thr"] = 100
        gen_mw_mask.args["shift"] = 1
        pre_transform = DataTransformer(
            transform_list=[
                gen_normalized_adj_matrix,
                gen_mw_ppm,
                gen_mw_mask,
                gen_knn_graph,
                gen_knn_distance,
                gen_radius_graph,
                gen_radius_distance
            ],
            inplace=True,
            rm_sup_info=True
        )

        # ScaffoldSplitter()
        print("(7/7) Setting up InMemoryDataSet")
        smrt = SMRT(
            root=root,
            data_list=smrt,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            splitter=RandomSplitter(seed=20211227)
        )
        RDLogger.EnableLog('rdApp.info')

    else:
        print("The small molecule retention time (SMRT) dataset has already been processed.")
