# %%
import numpy as np
import torch
import torch.nn.functional as F

from ART.model.ARTNet import ARTNet
from ART.Dataset.SELF_LAB import *
from ART.GenAttr import *

from ax.service.ax_client import AxClient
from torch_geometric.data import DataLoader
from scipy.stats import spearmanr, pearsonr, tstd

# %%
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
        
        if epoch > 100 and valid_mse > 20:
            break
        
        epoch += 1
        if epoch % 10 == 0:
            if epoch == 10:
                print(">>> Phase:", data.phase[0], "<<<")
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
def load_SELF_LAB(root:str, phase:str, batch_size:int, shuffle:bool, k:int=5):
    pre_transform = AttrsGenPipeline(fun_lst=[GenMolAttrs(), GenAtomAttrs(k=k), GenBondAttrs()])
    train_dataset = SELF_LAB(root=root, phase=phase, split = "train", transform=None, pre_transform=pre_transform)
    valid_dataset = SELF_LAB(root=root, phase=phase, split = "valid", transform=None, pre_transform=pre_transform)
    test_dataset  = SELF_LAB(root=root, phase=phase, split = "test",  transform=None, pre_transform=pre_transform)
    
    print("===================================== Sample =====================================")
    print(f"Trains ({len(train_dataset):d}), Validation ({len(valid_dataset):d}), Test ({len(test_dataset):d})\n")
    
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader  = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=shuffle)
    return (train_loader, valid_loader, test_loader)

# %%
if __name__ == '__main__':
    BATCH_SIZE = 64
    train_loader, valid_loader, test_loader = load_SELF_LAB(root="./SELF_LAB_wo_tt", phase="NEG", k=3, batch_size=128, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ax_client = AxClient()
    ax_client.create_experiment(
        name="SMRT",
        parameters=[
            {
                "name": "lr",
                "type": "range",
                "bounds": [-6.5, -4.0],
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
            # {
            #     "name": "momentum",
            #     "type": "range",
            #     "bounds": [0.0, 1.0],
            #     "value_type": "float",
            #     "log_scale": False,
            # },
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

    n_trail = 100
    for _ in range(n_trail):
        parameters, trial_index = ax_client.get_next_trial()
        print(f"\nTrail ({trial_index+1:02d}/{n_trail:02d})")
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data=evaluate(
                k=3,
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
    trials_pars_df.to_csv("./Ax_Results/SELF_wo_tt_AdamW.csv")

    best_parameters, values = ax_client.get_best_parameters()
    print(best_parameters)
    means, covariances = values 

    print("\n")
    print("================================= Best Parameters =================================")
    print(f'mean: {means["valid_mse"]:5.3f}, epoch: {int(means["epoch"]):02d}')
    print(best_parameters)