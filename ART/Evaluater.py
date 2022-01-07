import gc
import torch
import numpy as np

from .ParSet import AttentiveFPPars, PredictorPars
from .ARTNet import ARTNet
from .funcs import json_snapshot_to_doc

from ax.service.ax_client import AxClient
from collections import OrderedDict
from pymongo.collation import Collation
from torch_geometric.profile import get_gpu_memory_from_nvidia_smi
from torch_geometric.loader import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from typing import List, Union


class Evaluater():
    def __init__(
            self,
            train_loader: Union[DataLoader, List[Data]],
            valid_loader: Union[DataLoader, List[Data]],
            learning_rate: Union[float, int],
            embd_lyr_pars_dict: OrderedDict,
            afp_mdl_pars: AttentiveFPPars,
            prdctr_lyr_pars: PredictorPars,
            ax_client: AxClient,
            weight_decay: Union[float, int] = 0,
            momentum: Union[float, int] = 0,
            device: str = "cpu",
            max_epoch: int = 100,
            mngdb_snapshot: Union[None, Collation] = None,
            tqdm_ncols: int = 70
            ) -> None:
        self.device = device
        self.ax_client = ax_client
        self.mngdb_snapshot = mngdb_snapshot
        self.tqdm_ncols = tqdm_ncols
        self.max_epoch = max_epoch

        # Data
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Hyperparameters
        # Network architecture
        self.model = ARTNet(
            embd_lyr_pars_dict=embd_lyr_pars_dict,
            afp_mdl_pars=afp_mdl_pars,
            prdctr_lyr_pars=prdctr_lyr_pars
        ).to(device)
        # Learner
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=10**learning_rate,
            weight_decay=10**weight_decay
        )
        # Criterian
        self.criterian = torch.nn.MSELoss(
            reduction='sum'
        ).to(device=device)

    def print_gpu_ram_usage(self):
        print(f"   > Using device: {self.device}.")
        if self.device.type == 'cuda':
            free_gpu_mem, used_gpu_mem = get_gpu_memory_from_nvidia_smi()
            gpu_prec = round(used_gpu_mem / free_gpu_mem * 100, 2)
            print(f"    > GPU Memory: {used_gpu_mem}/{free_gpu_mem} ({gpu_prec}%) megabytes")

    def train(self):
        self.model.train()
        gc.collect()
        torch.cuda.empty_cache()
        train_loss = torch.tensor(
                0, dtype=torch.float64, requires_grad=False
            ).to(device=self.device)
        train_graphs = 0
        self.print_gpu_ram_usage()
        for data in tqdm(
                self.train_loader, total=len(self.train_loader),
                desc="    - T", ncols=self.tqdm_ncols):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            weight = torch.sqrt(1.0/data.n_tt).double()
            outputs = self.model(data).squeeze()
            targets = data.y
            # Optimize model according to weighted loss
            loss = self.criterian(outputs*weight, targets*weight)
            loss.backward()
            self.optimizer.step()
            # MSE
            with torch.no_grad():
                train_loss += self.criterian(outputs, targets)
                train_graphs += data.num_graphs
            # free memory
            gc.collect()
        torch.cuda.empty_cache()
        # return float(train_loss.to("cpu")/train_graphs)
        return float(torch.sqrt(train_loss / train_graphs).to("cpu"))

    def validate(self):
        self.model.eval()
        torch.cuda.empty_cache()
        valid_err = torch.zeros(
            len(self.valid_loader),
            dtype=torch.float64,
            requires_grad=False).to(device=self.device)
        with torch.no_grad():
            for idx, data in enumerate(tqdm(
                    self.valid_loader, total=len(self.valid_loader),
                    desc="    - V", ncols=self.tqdm_ncols)):
                data = data.to(self.device)
                outputs = self.model(data).squeeze()
                targets = data.y
                valid_err[idx] = self.criterian(outputs, targets) / data.num_graphs
                gc.collect()

        torch.cuda.empty_cache()
        valid_err = torch.sqrt(valid_err)
        valid_err_mu = float(torch.mean(valid_err).to("cpu"))
        valid_err_std = float(torch.std(valid_err).to("cpu"))
        return (valid_err_mu, valid_err_std)

    def evaluate(self):
        self.model.eval()
        rt_r = np.zeros(len(self.valid_loader))
        rt_rho = np.zeros(len(self.valid_loader))
        with torch.no_grad():
            for idx, data in enumerate(tqdm(
                    self.valid_loader, total=len(self.valid_loader),
                    desc="Summarize Results", ncols=self.tqdm_ncols)):
                data = data.to(self.device)
                rt_true_batch = self.model(data).squeeze().to("cpu").tolist()
                rt_prdt_batch = data.y.to("cpu").tolist()
                r, _ = pearsonr(rt_true_batch, rt_prdt_batch)
                rho, _ = spearmanr(rt_true_batch, rt_prdt_batch)
                rt_r[idx] += r
                rt_rho[idx] += rho
                gc.collect()
        torch.cuda.empty_cache()
        return {
            "pearson_r": {
                "mu": np.mean(rt_r),
                "std": np.std(rt_r)
            },
            "spearman_rho": {
                "mu": np.mean(rt_rho),
                "std": np.std(rt_rho)
            }
        }

    def run(self):
        epoch = 0
        min_mse = 1e10
        min_std = 1e10
        counter = 0
        train_mse = 1e10
        valid_mse = 1e10

        if self.device.type == 'cuda':
            print(f"   > Device name: {torch.cuda.get_device_name(0)}")
        
        divider_len = 50
        print("=" * divider_len + " Start " + "=" * divider_len)
        while epoch < self.max_epoch:
            train_mse = self.train()

            if train_mse is None:
                valid_mse = 1e10
                train_mse = 1e10
                break

            valid_mse, valid_std = self.validate()

            # Early stop
            if min_mse >= valid_mse:
                min_mse = valid_mse
                min_std = valid_std
                counter = 0
            else:
                if epoch > 25:
                    counter += 1
                if counter > 15:
                    break

            if epoch > 10 and valid_mse > 300:
                break

            if epoch > 35 and valid_mse > 120:
                break

            epoch += 1
            print(f'> Epoch: {epoch:05d}, Loss (Train): {round(train_mse, 3):5.3f}, Loss (Validation): {round(valid_mse, 3):5.3f}, flag: {counter:02d}')

        corr = self.evaluate()

        if self.mngdb_snapshot is not None:
            json_snapshot = self.ax_client.to_json_snapshot()
            doc = json_snapshot_to_doc(json_snapshot, compress=True)
            self.mngdb_snapshot.insert_one(doc)

        print("=" * divider_len + "= End =" + "=" * divider_len)
        return {
            "train_mse": (train_mse, 0),
            "valid_mse": (min_mse, min_std),
            "epoch": (epoch, 0),
            "r": (corr["pearson_r"]["mu"], corr["pearson_r"]["std"]),
            "rho": (corr["spearman_rho"]["mu"], corr["spearman_rho"]["std"])
        }
