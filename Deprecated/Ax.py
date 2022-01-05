# %%
import sys
ART_LAB_PATH = "./ART"
if ART_LAB_PATH not in sys.path:
    sys.path.append(ART_LAB_PATH)

import numpy as np
import torch
import torch.nn.functional as F

from ART.model.ARTNet import ARTNet
from ART.Dataset.SMRT import *
from ART.GenAttr import *
from ART.DataSplitter import CidRandomSplitter

from ax.service.ax_client import AxClient
from torch_geometric.data import DataLoader
from scipy.stats import spearmanr, pearsonr, tstd

def load_SMRT(root:str, batch_size:int, shuffle:bool, transform=None, pre_transform=None, pre_filter=None, n_jobs:int=12, splitter = None, tautomer:int=-1):
    train_dataset = SMRT(root=root, split="train", transform=transform, pre_transform=pre_transform, pre_filter=pre_filter, n_jobs=n_jobs, splitter=splitter, tautomer=tautomer)
    valid_dataset = SMRT(root=root, split="valid", transform=transform, pre_transform=pre_transform, pre_filter=pre_filter, n_jobs=n_jobs, splitter=splitter, tautomer=tautomer)
    test_dataset  = SMRT(root=root, split="test",  transform=transform, pre_transform=pre_transform, pre_filter=pre_filter, n_jobs=n_jobs, splitter=splitter, tautomer=tautomer)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader  = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=shuffle)
    return (train_loader, valid_loader, test_loader)


def evaluate(k:int, lr:float, weight_decay:float, train_loader:DataLoader, valid_loader:DataLoader, device:str, 
    afp_timesteps:int, afp_hcnl:int, afp_output_cnl:int, apf_num_layers:int, dropout:float=0.1):
    model = ARTNet(
        k=k, afp_timesteps=afp_timesteps, afp_hcnl=afp_hcnl,
        afp_output_cnl=afp_output_cnl, apf_num_layers=apf_num_layers,
        dropout=dropout
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = F.mse_loss

    # for training set
    epoch = 0
    min_mse = 1e10
    min_std = 1e10
    counter = 0
    train_mse = 1e10
    valid_mse = 1e10
    print("===================================== Start =====================================")
    while epoch < 300:
        total_loss = 0.0
        total_examples = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data, data.batch)
            target = data.rt.unsqueeze(1)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * data.num_graphs
            total_examples += data.num_graphs
            torch.cuda.empty_cache()
        train_mse = np.sqrt(total_loss / total_examples)
        
        if train_mse is None:
            valid_mse = 1e10
            train_mse = 1e10
            break
        # for validation set
        valid_err = []
        for data in valid_loader:
            data = data.to(device)
            outputs = model(data, data.batch)
            target = data.rt.unsqueeze(1)
            valid_err += ((outputs - target)**2).squeeze().to("cpu").tolist()
        torch.cuda.empty_cache()
        valid_err = np.array(valid_err)
        valid_mse = np.sqrt(sum(valid_err)/len(valid_err))
        
        if min_mse >= valid_mse:
            min_mse = valid_mse
            min_std = np.std(valid_err)
            counter = 0
        else:
            if epoch > 125:
                counter += 1
            if counter > 20:
                # print(f'Epoch: {epoch:05d}, Loss (Train): {round(train_mse, 3):5.3f}, Loss (Validation): {round(valid_mse, 3):5.3f}, flag: {counter: 02d}')
                # print('------ end ------')
                break
        
        if epoch > 100 and valid_mse > 500:
            break
        
        if epoch > 125 and valid_mse > 250:
            break
        
        epoch += 1
        if epoch % 10 == 0:
            print(f'    > Epoch: {epoch:05d}, Loss (Train): {round(train_mse, 3):5.3f}, Loss (Validation): {round(valid_mse, 3):5.3f}, flag: {counter:02d}')
    
    rt_true = []
    rt_prdt = []
    rt_r    = []
    rt_rho  = []

    for data in valid_loader:
        data = data.to(device)
        outputs = model(data, data.batch)
        target = data.rt.unsqueeze(1)
        rt_true_batch = target.squeeze().to("cpu").tolist()
        rt_prdt_batch = outputs.squeeze().to("cpu").tolist()
        r, _    = spearmanr(rt_true_batch, rt_prdt_batch)
        rho, _  = pearsonr(rt_true_batch, rt_prdt_batch)
        rt_r    += [r]
        rt_rho  += [rho]
        rt_true += rt_true_batch
        rt_prdt += rt_prdt_batch
    torch.cuda.empty_cache()

    rt_true = np.array(rt_true)
    rt_prdt = np.array(rt_prdt)
    r_std   = tstd(rt_r)
    rho_std = tstd(rt_rho)
    r, _    = spearmanr(rt_true, rt_prdt)
    rho, _  = pearsonr(rt_true, rt_prdt)
    
    print("====================================== End ======================================")
    
    return {
        "train_mse": (train_mse, 0),
        "valid_mse": (min_mse, min_std), 
        "epoch": (epoch, None),
        "r":(r, r_std),
        "rho":(rho, rho_std)
    }

# %%
if __name__ == '__main__':
    BATCH_SIZE = 1024

    gen_mol_attr  = GenMolAttrs()
    gen_atom_attr = GenAtomAttrs(k=5)
    gen_bond_attr = GenBondAttrs()
    pre_transform = AttrsGenPipeline(fun_lst=[gen_mol_attr, gen_atom_attr, gen_bond_attr])

    splitter = CidRandomSplitter()

    train_loader, valid_loader, test_loader = load_SMRT(root="./SMRT_smi", batch_size=BATCH_SIZE, pre_transform=pre_transform, shuffle=True, n_jobs=12, splitter=splitter, tautomer=-1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ax_client = AxClient()
    ax_client.create_experiment(
        name="SMRT",
        parameters=[
            {
                "name": "lr",
                "type": "range",
                "bounds": [-7, -3],
                "value_type": "float",
                "log_scale": False,
            },
            {
                "name": "weight_decay",
                "type": "choice",
                "values": [-3.0, -2.5, -2.0, -1.5, -1.0],
                "value_type": "float",
                "log_scale": False,
            },
            {
                "name": "afp_timesteps",
                "type": "choice",
                "values": [1, 2],
                "value_type": "int",
                "log_scale": False,
            },
            {
                "name": "afp_hcnl",
                "type": "choice",
                "values": [256, 512, 1024],
                "value_type": "int",
                "log_scale": False,
            },
            {
                "name": "afp_output_cnl",
                "type": "choice",
                "values": [512, 1024, 2048],
                "value_type": "int",
                "log_scale": False,
            },
            {
                "name": "apf_num_layers",
                "type": "choice",
                "values": [1, 2, 3],
                "value_type": "int",
                "log_scale": False,
            },
        ],
        objective_name="valid_mse",
        minimize=True
    )

    n_trail = 25
    for _ in range(n_trail):
        parameters, trial_index = ax_client.get_next_trial()
        print(f"\nTrail ({trial_index+1:02d}/{n_trail:02d})")
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data=evaluate(
                k=5,
                lr=10**parameters.get("lr"),
                weight_decay=10**parameters.get("weight_decay"),
                # momentum=parameters.get("momentum"),
                train_loader=train_loader,
                valid_loader=valid_loader,
                device=device,
                afp_timesteps=parameters.get("afp_timesteps"),
                afp_hcnl=parameters.get("afp_hcnl"),
                afp_output_cnl=parameters.get("afp_output_cnl"),
                apf_num_layers=parameters.get("apf_num_layers"),
                dropout=0.1
            )
        )
    
    trials_pars_df = ax_client.get_trials_data_frame()
    trials_pars_df.to_csv("./Ax_Results/SMRT_wo_tt_AdamW.csv")

    best_parameters, values = ax_client.get_best_parameters()
    print(best_parameters)
    means, covariances = values 

    print("\n")
    print("================================= Best Parameters =================================")
    print(f'mean: {means["valid_mse"]:5.3f}, epoch: {int(means["epoch"]):02d}')
    print(best_parameters)

# %%
# BATCH_SIZE = 1024
# train_loader, valid_loader, test_loader = load_SMRT("./SMRT_w_tt", batch_size=BATCH_SIZE, shuffle=True)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = Net(k=5, afp_timesteps=2, afp_hcnl=512, afp_output_cnl=1024, dropout=0.1).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0001)
# criterion = F.mse_loss

# epoch     = 0
# min_err   = 1e10
# counter   = 0
# train_mse = 1e10
# valid_mse = 1e10
# valid_std = 0
# print("Start")
# while epoch < 2:
#     total_loss = total_examples = 0
#     for data in train_loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         outputs = model(data, data.batch)
#         target = data.rt.unsqueeze(1)
#         loss = criterion(outputs, target)
#         loss.backward()
#         optimizer.step()
#         total_loss += float(loss) * data.num_graphs
#         total_examples += data.num_graphs
#     train_mse = np.sqrt(total_loss / total_examples)
        
#     if train_mse is None:
#         valid_mse = 1e10
#         train_mse = 1e10
#         break
#     # # for validation set
#     valid_err = []
#     for data in valid_loader:
#         data = data.to(device)
#         outputs = model(data, data.batch)
#         target = data.rt.unsqueeze(1)
#         valid_err += ((outputs - target)**2).squeeze().to("cpu").tolist()
#     torch.cuda.empty_cache()
#     valid_err = np.array(valid_err)
#     valid_mse = np.sqrt(sum(valid_err)/len(valid_err))
    
#     if min_err >= valid_mse:
#         min_err = valid_mse
#         counter = 0
#     else:
#         if epoch > 125:
#             counter += 1
#         if counter > 30:
#             # print(f'Epoch: {epoch:05d}, Loss (Train): {round(train_mse, 3):5.3f}, Loss (Validation): {round(valid_mse, 3):5.3f}, flag: {counter: 02d}')
#             # print('------ end ------')
#             break
#     epoch += 1
#     print(f'\t > Epoch: {epoch:05d}, Loss (Train): {round(train_mse, 3):5.3f}, Loss (Validation): {round(valid_mse, 3):5.3f}, flag: {counter: 02d}')
#     # print(f'\t > Epoch: {epoch:05d}, Loss (Train): {round(train_mse, 3):5.3f}, flag: {counter: 02d}')
# print("End")
# # %%

# import json
# path = [
#     "./SMRT_w_tt/json/SMRT_train.json",
#     "./SMRT_w_tt/json/SMRT_valid.json",
#     "./SMRT_w_tt/json/SMRT_test.json"
# ]

# train_dataset = SMRT(root = "./SMRT_smi", split = "train", transform=None, pre_transform=GenAtomAndBondFeatures(k=5))
# valid_dataset = SMRT(root = "./SMRT_smi", split = "valid", transform=None, pre_transform=GenAtomAndBondFeatures(k=5))
# test_dataset  = SMRT(root = "./SMRT_smi", split = "test",  transform=None, pre_transform=GenAtomAndBondFeatures(k=5))

# def saveToJson(dataset, path):
#     data_list = []
#     for data in dataset:
#         mol = {}
#         for k in data["keys"]:
#             item = data[k]
#             if isinstance(item, torch.Tensor):
#                 if item.dim() == 1 and item.shape[0] == 1:
#                     item = item.tolist()[0]
#                 else:
#                     item = item.tolist()
#             mol[k] = item
#         data_list.append(mol)

#     with open(path, "w", newline="") as jsonfile:
#         json.dump(data_list, fp=jsonfile)