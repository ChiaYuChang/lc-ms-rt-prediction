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
from ART.DataTransformer.DataTransformer import DataTransformer, gen_normalize_adj_matrix
from ART.DataTransformer.DataTransformer import gen_knn_graph, gen_knn_distance
    
from pathos.multiprocessing import ProcessingPool
from rdkit import RDLogger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm


if __name__ == '__main__':
    n_jobs = 40
    RDLogger.DisableLog('rdApp.*')

    mol_reader = SMRTSdfReader(
        file_path="./Data/SMRT/raw/SMRT_dataset.sdf"
    )

    parallel_mol_reader = ParallelMolReader(
        n_jobs=n_jobs,
        inplace=False
    )

    featurizer = Featurizer(
        feature_set=DefaultFeatureSet(),
        include_coordinates=True
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
    
    print("Setting up InMemoryDataSet")
    smrt = SMRT(root="./SMRT/", data_list=smrt, splitter=RandomSplitter(seed=20211227))

    RDLogger.EnableLog('rdApp.info')

    transform = DataTransformer(
        transform_list=[
            gen_normalize_adj_matrix,
            gen_knn_graph,
            gen_knn_distance
        ]
    )

    smrt = SMRT(
        root="./SMRT/",
        data_list=None,
        transform=transform,
        splitter=None
    )
    smrt_loader = DataLoader(
        dataset=smrt,
        batch_size=512,
        shuffle=True
    )

    gcn_par = GCNLayerPar(
        in_channels=smrt[0].node_attr.shape[1],
        out_channels=64,
        dropout=0.1,
        relu=True,
        batch_norm=True
    )

    gcn_lyr = GraphConvLayer(
        in_channels=gcn_par.in_channels,
        out_channels=gcn_par.out_channels,
        dropout=gcn_par.dropout,
        relu=gcn_par.relu,
        batch_norm=gcn_par.batch_norm
    )

    data = smrt[0]
    gcn_lyr.forward(data.normalize_adj_matrix, data.node_attr)

