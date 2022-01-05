# %%
from ART.DataSet import SMRT, PredRet
from ART.DataSplitter import RandomSplitter
from ART.Featurizer.FeatureSet import DefaultFeatureSet
from ART.Featurizer.Featurizer import Featurizer, ParallelFeaturizer
from ART.FileReaders import ParallelMolReader
from ART.FileReaders import CsvReader, SMRTSdfReader
from ART.model.KensertGCN.GraphConvLayer import GraphConvLayer
from ART.model.KensertGCN.model import KensertGCN
from ART.ParSet import LinearLayerPars
from ART.DataTransformer.DataTransformer import DataTransformer
from ART.DataTransformer.Transforms import gen_knn_graph, gen_knn_distance, gen_normalize_adj_matrix
from ART.funcs import check_has_processed, array_to_tensor

from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool
from rdkit import RDLogger
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# %%
if __name__ == '__main__':
    # %%
    root = "./Data/SMRT"
    raw_file_names = "SMRT_dataset.sdf"
    
    # %%
    if not(check_has_processed(root=root, raw_file_names=raw_file_names)):
        print(f"Calculating descriptors")
        n_jobs = cpu_count() // 4 * 3
        print(f"Using {n_jobs} cores for preprocessing")
        
        RDLogger.DisableLog('rdApp.*')

        print("Setting up Reader and Featurizer")
        mol_reader = SMRTSdfReader(
            file_path="/".join((root, "raw", raw_file_names))
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
        
        print("Reading File")
        smrt = parallel_mol_reader(mol_reader=mol_reader, pool=pool)

        print("Featurizing")
        smrt = parallel_featureizer(file_reader=smrt, pool=pool)
        
        print("Converting np.array to torch.tensor")
        for data in tqdm(smrt, desc="converting"):
            array_to_tensor(data=data, in_place=True)

        print("Setting up InMemoryDataSet")
        smrt = SMRT(
            root=root,
            data_list=smrt,
            transform=None,
            splitter=RandomSplitter(seed=20211227)
        )
        RDLogger.EnableLog('rdApp.info')

    # %%
    transform = DataTransformer(
        transform_list=[
            gen_normalize_adj_matrix,
            gen_knn_graph,
            gen_knn_distance
        ]
    )

    # %%
    smrt = SMRT(
        root=root,
        transform=transform
    )

    # %%
    smrt_loader = DataLoader(
        dataset=smrt,
        batch_size=512,
        shuffle=True
    )

    # %%
    # gcn_par = GCNLayerPar(
    #     in_channels=smrt[0].node_attr.shape[1],
    #     out_channels=64,
    #     dropout=0.1,
    #     relu=True,
    #     batch_norm=True
    # )

    # gcn_lyr = GraphConvLayer(
    #     in_channels=gcn_par.in_channels,
    #     out_channels=gcn_par.out_channels,
    #     dropout=gcn_par.dropout,
    #     relu=gcn_par.relu,
    #     batch_norm=gcn_par.batch_norm
    # )

    # data = smrt[0]
    # gcn_lyr.forward(data.normalize_adj_matrix, data.node_attr)
