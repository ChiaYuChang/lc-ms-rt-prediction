import torch
import gc
import psutil
import pymongo
import numpy as np

from ART.funcs import json_snapshot_to_doc

from scipy.stats import spearmanr, pearsonr
from torch import nn
from torch import optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.profile import get_gpu_memory_from_nvidia_smi
from tqdm import tqdm
from typing import Dict, List, NamedTuple, Optional, Tuple, Union


class ModelEvaluator():

    def __init__(
            self,
            model: nn.Module,
            optimizer: optim.Optimizer,
            loss: nn.Module,
            train_loader: Union[DataLoader, List[Data]], 
            valid_loader: Union[DataLoader, List[Data]],
            device: torch.device = torch.device("cpu"),
            weight: Optional[List[float]] = torch.tensor(1),
            max_epoch: int = 100,
            tqdm_ncols: int = 70) -> None:
        
        self._model = model
        self._optim = optimizer
        self._loss = loss
        self._weight = weight
        self._loader = {
            "train": train_loader,
            "valid": valid_loader
        }
        self._max_epoch = max_epoch
        self._tqdm_ncols = tqdm_ncols
        self._device = device
        self.to()
        return None

    @property
    def device(self):
        return self._device

    @property
    def model(self):
        return self._model
    
    @property
    def optimizer(self):
        return self._optim

    @property
    def loss(self):
        return self._loss

    @property
    def model(self):
        return self._model

    @property
    def weight(self):
        return self._weight

    @property
    def max_epoch(self):
        return self._max_epoch

    @property
    def train_loader(self):
        return self._loader["train"]
    
    @property
    def valid_loader(self):
        return self._loader["valid"]

    def loader(self, slice: str):
        return self._loader[slice]

    def to(self, device: Optional[torch.device] = None) -> None:
        if device is not None:
            self._device = device
        print(f"Moving parameters to {self._device.type}:{self._device.index}")
        print("   > model ... ", end="")
        self._model.to(self._device)
        print("OK")
        print("   > loss .... ", end="")
        self._loss.to(self._device)
        print("OK")
        return None
    
    def ram_cleanup(self, console:bool = False) -> None:
        gc.collect()
        torch.cuda.empty_cache()
        if console:
            print(
                " - RAM usage: " + "; ".join([
                    f"CPU : {self.cpu_ram_usage():05.2f}%",
                    f"SWAP: {self.swap_usage():05.2f}%",
                    f"GPU : {self.gpu_ram_usage():05.2f}%"
                ])
            )

    def swap_usage(self) -> float:
        return psutil.swap_memory().percent

    def cpu_ram_usage(self) -> float:
        return psutil.virtual_memory().percent

    def gpu_ram_usage(self) -> float:
        if self.device.type == 'cuda':
            free_gpu_mem, used_gpu_mem = get_gpu_memory_from_nvidia_smi()
            return round(used_gpu_mem / free_gpu_mem * 100, 2)
        else:
            return 0

    def train(self) -> Tuple[float, float]:
        self.model.train()
        self.ram_cleanup(console=True)
        train_loss = torch.empty(
            len(self.train_loader),
            dtype=torch.float64,
            requires_grad=False
            ).to(device=self.device)

        for idx, data in enumerate(tqdm(
                self.train_loader, total=len(self.train_loader),
                desc=" - T", ncols=self._tqdm_ncols)):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            outputs = self.model(data)
            targets = data.y
            
            # Optimize model according to weighted loss
            loss = self.loss(outputs*self.weight, targets*self.weight)
            loss.backward()
            self.optimizer.step()
            
            # MSE
            with torch.no_grad():
                train_loss[idx] = self.loss(outputs, targets)
        
        # free memory
        train_loss_mu = float(torch.mean(train_loss).to("cpu"))
        if len(train_loss) > 1:
            train_loss_std = float(torch.std(train_loss).to("cpu"))
        else:
            train_loss_std = 0.0
        
        self.ram_cleanup()
        return (train_loss_mu, train_loss_std)

    def validate(self) -> Tuple[float, float]:
        self.model.eval()
        self.ram_cleanup()
        valid_loss = torch.empty(
            len(self.valid_loader),
            dtype=torch.float64,
            requires_grad=False
            ).to(device=self.device)

        with torch.no_grad():
            for idx, data in enumerate(tqdm(
                    self.valid_loader, total=len(self.valid_loader),
                    desc=" - V", ncols=self._tqdm_ncols)):
                data = data.to(self.device)
                outputs = self.model(data)
                targets = data.y
                valid_loss[idx] = self.loss(outputs, targets)
        
        valid_loss_mu = float(torch.mean(valid_loss).to("cpu"))
        if len(valid_loss) > 1:
            valid_loss_std = float(torch.std(valid_loss).to("cpu"))
        else:
            valid_loss_std = 0.0
        
        self.ram_cleanup()
        return (valid_loss_mu, valid_loss_std)

    def evaluate(self):
        mse_loss = nn.MSELoss(reduction='mean')
        l1_loss = nn.L1Loss(reduction='mean')
        
        self.model.eval()
        self.ram_cleanup()
        rt_r = np.zeros(len(self.valid_loader))
        
        rt_rho = np.zeros(len(self.valid_loader))
        
        rmse = torch.zeros(
            len(self.valid_loader),
            dtype=torch.float64,
            requires_grad=False).to(device=self.device)
        
        mae = torch.zeros(
            len(self.valid_loader),
            dtype=torch.float64,
            requires_grad=False).to(device=self.device)
        
        with torch.no_grad():
            for idx, data in enumerate(tqdm(
                    self.valid_loader, total=len(self.valid_loader),
                    desc=" - S", ncols=self._tqdm_ncols)):
                data = data.to(self.device)
                
                targets = data.y
                outputs = self.model(data)
                
                rmse[idx] = torch.sqrt(mse_loss(targets, outputs))
                mae[idx] = l1_loss(targets, outputs)
                
                targets = targets.to("cpu").numpy()
                outputs = outputs.squeeze().to("cpu").numpy()
                r, _ = pearsonr(outputs, targets)
                rho, _ = spearmanr(outputs, targets)
                rt_r[idx] = r
                rt_rho[idx] = rho
        
        rmse = rmse.to("cpu").numpy()
        mae = mae.to("cpu").numpy()
        self.ram_cleanup()
        return {
            "rmse": {
                "mu": np.mean(rmse),
                "std": np.std(rmse)
            },
            "mae": {
                "mu": np.mean(mae),
                "std": np.std(mae)
            },
            "pearson_r": {
                "mu": np.mean(rt_r),
                "std": np.std(rt_r)
            },
            "spearman_rho": {
                "mu": np.mean(rt_rho),
                "std": np.std(rt_rho)
            }
        }

    def run(self, trial_index: Optional[int] = None):
        epoch = 0
        min_loss_mu = 1e10
        min_loss_std = 1e10
        counter = 0
        train_loss_mu = 1e10
        valid_loss_mu = 1e10
        valid_loss_std = 1e10

        if self.device.type == 'cuda':
            print(f"   > Device name: {torch.cuda.get_device_name(0)}")

        divider_len = 50
        print("=" * divider_len + " Start " + "=" * divider_len)

        while epoch < self._max_epoch:
            if trial_index is None:
                print(f"Epoch: {epoch:05d}")
            else:
                print(f"Trial: {trial_index:02d} Epoch: {epoch:05d}")
            train_loss, _ = self.train()

            if train_loss is torch.nan or train_loss is None:
                print("produce nan during calculation")
                valid_loss_mu = 1e10
                train_loss_mu = 1e10
                break
            else:
                train_loss_mu = train_loss
            
            valid_loss_mu, valid_loss_std = self.validate()
            # Early stop
            if min_loss_mu >= valid_loss_mu:
                min_loss_mu = valid_loss_mu
                min_loss_std = valid_loss_std
                counter = 0
            else:
                if (epoch > 50): # dropout
                    counter += 1
                if counter > 25:
                    break

            epoch += 1

            evaluate_result = self.evaluate()
            print(
                f" - Loss (Train): {train_loss_mu:5.3f}" +\
                f"; Loss (Valid): {valid_loss_mu:5.3f}"
            )
            print(
                f" - RMSE: {evaluate_result['rmse']['mu']:.3f}" +\
                f"; MAE: {evaluate_result['mae']['mu']:.3f}" +\
                f"; Counter: {counter:02d}"
            )

        evaluate_result = self.evaluate()

        print("=" * divider_len + "= End =" + "=" * divider_len)
        return {
            "train_loss": (train_loss_mu, 0.00),
            "valid_loss": (min_loss_mu, min_loss_std),
            "epoch": (epoch, 0),
            "rmse": (
                evaluate_result["rmse"]["mu"],
                evaluate_result["rmse"]["std"]
            ),
            "mae": (
                evaluate_result["mae"]["mu"],
                evaluate_result["mae"]["std"]
            ),
            "r": (
                evaluate_result["pearson_r"]["mu"],
                evaluate_result["pearson_r"]["std"]),
            "rho": (
                evaluate_result["spearman_rho"]["mu"],
                evaluate_result["spearman_rho"]["std"])
        }
