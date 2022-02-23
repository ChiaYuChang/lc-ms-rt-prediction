import imp
import torch
import gc
import psutil
import pymongo
import numpy as np

from ART.funcs import json_snapshot_to_doc

from scipy.stats import spearmanr, pearsonr
from torch import nn
from torch import optim
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.profile import get_gpu_memory_from_nvidia_smi
from tqdm import tqdm
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union


class ModelEvaluator():

    def __init__(
            self,
            model: nn.Module,
            optimizer: optim.Optimizer,
            loss: nn.Module,
            train_loader: Union[DataLoader, List[Data]], 
            valid_loader: Union[DataLoader, List[Data]],
            target_transform: Optional[Callable[[Data], Tensor]] = lambda x: x.y,
            device: torch.device = torch.device("cpu"),
            count_down_thr: Optional[int] = 50,
            max_epoch: Optional[int] = 100,
            tqdm_ncols: Optional[int] = 70) -> None:
        
        self._model = model
        self._optim = optimizer
        self._loss = loss
        self._loader = {
            "train": train_loader,
            "valid": valid_loader
        }
        self._max_epoch = max_epoch
        self._tqdm_ncols = tqdm_ncols
        self._device = device
        self._count_down_thr = count_down_thr
        self._target_transform = target_transform
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
    def max_epoch(self):
        return self._max_epoch

    @property
    def train_loader(self):
        return self._loader["train"]
    
    @property
    def valid_loader(self):
        return self._loader["valid"]

    @property
    def target_transform(self):
        return self._target_transform

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
            targets = self.target_transform(data)
            
            # Optimize model according to weighted loss
            loss = self.loss(outputs, targets)
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
                targets = self.target_transform(data)
                valid_loss[idx] = self.loss(outputs, targets)
        
        valid_loss_mu = float(torch.mean(valid_loss).to("cpu"))
        if len(valid_loss) > 1:
            valid_loss_std = float(torch.std(valid_loss).to("cpu"))
        else:
            valid_loss_std = 0.0
        
        self.ram_cleanup()
        return (valid_loss_mu, valid_loss_std)

    def evaluate(self) -> Dict:
        return None

    def print_progress(
            self, train_result, valid_result, evaluate_result,
            min_loss, counter, epoch) -> None:
        print(
            f" - Loss (Train): {train_result[0]:5.3f}" +\
            f"; Loss (Valid): {valid_result[0]:5.3f}" +\
            f"; Loss (Min): {min_loss[0]:5.3f}"
        )
        return None

    def extract_result(
            self, train_result, valid_result, evaluate_result,
            min_loss, counter, epoch) -> Dict:
        raise NotImplementedError

    def run(self, trial_index: Optional[int] = None):
        epoch = 0
        counter = 0
        min_loss_mu  = 1e10
        min_loss_std = 1e10

        if self.device.type == 'cuda':
            print(f"   > Device name: {torch.cuda.get_device_name(0)}")

        divider_len = 50
        print("=" * divider_len + " Start " + "=" * divider_len)

        while epoch < self._max_epoch:
            if trial_index is None:
                print(f"Epoch: {epoch:05d}")
            else:
                print(f"Trial: {trial_index:02d} Epoch: {epoch:05d}")

            # Train
            train_loss_mu, train_loss_std = self.train()
            if train_loss_mu is torch.nan or train_loss_mu is None:
                print("produce nan during calculation")
                valid_loss_mu = 1e10
                train_loss_mu = 1e10
                break
            
            # Validate
            valid_loss_mu, valid_loss_std = self.validate()

            if min_loss_mu > 1e8 and epoch > 10:
                break

            # Early stop
            if min_loss_mu >= valid_loss_mu:
                min_loss_mu = valid_loss_mu
                min_loss_std = valid_loss_std
                counter = 0
            else:
                if (epoch > self._count_down_thr): # dropout
                    counter += 1
                if counter > 25:
                    break

            epoch += 1

            # Evaluate
            evaluate_result = self.evaluate()
            
            # Show progress
            self.print_progress(
                train_result=(train_loss_mu, train_loss_std),
                valid_result=(valid_loss_mu, valid_loss_std),
                evaluate_result=evaluate_result,
                min_loss=(min_loss_mu, min_loss_std),
                counter=counter,
                epoch=epoch)

        evaluate_result = self.evaluate()

        print("=" * divider_len + "= End =" + "=" * divider_len)
        return self.extract_result(
                train_result=(train_loss_mu, train_loss_std),
                valid_result=(valid_loss_mu, valid_loss_std),
                evaluate_result=evaluate_result,
                min_loss=(min_loss_mu, min_loss_std),
                counter=counter,
                epoch=epoch)


class RegressionModelEvaluator(ModelEvaluator):
    __name__ = "RegressionModelEvaluator"

    def evaluate(self) -> Dict:
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
                
                targets = self.target_transform(data)
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
            "rmse" : (np.mean(rmse), np.std(rmse)),
            "mae" : (np.mean(mae), np.std(mae)),
            "pearson_r" : (np.mean(rt_r), np.std(rt_r)),
            "spearman_rho": (np.mean(rt_rho), np.std(rt_rho))
        }
    
    def print_progress(
            self, train_result, valid_result, evaluate_result,
            min_loss, counter, epoch) -> None:
        
        super().print_progress(
            train_result, valid_result, evaluate_result,
            min_loss, counter, epoch)
        
        print(
            f" - RMSE: {evaluate_result['rmse'][0]:.3f}" +\
            f"; MAE: {evaluate_result['mae'][0]:.3f}" +\
            f"; Counter: {counter:02d}"
        )
        return None

    def extract_result(
           self, train_result, valid_result, evaluate_result,
            min_loss, counter, epoch) -> Dict:
        return {
            "train_loss": train_result,
            "valid_loss": min_loss,
            "epoch": (epoch, 0),
            "rmse": evaluate_result["rmse"],
            "mae": evaluate_result["mae"],
            "r": evaluate_result["pearson_r"],            
            "rho": evaluate_result["spearman_rho"]
        }


class ClassificationModelEvaluator(ModelEvaluator):

    def __init__(
            self,
            model: nn.Module,
            optimizer: optim.Optimizer,
            loss: nn.Module,
            train_loader: Union[DataLoader, List[Data]],
            valid_loader: Union[DataLoader, List[Data]],
            target_transform: Optional[Callable[[Data], Tensor]]=lambda x: x.y_cat[:, 0],
            acc_k_top: Optional[int] = 3,
            device: torch.device = torch.device("cpu"),
            count_down_thr: Optional[int] = 50,
            max_epoch: Optional[int] = 100,
            tqdm_ncols: Optional[int] = 70) -> None:
        
        super().__init__(
            model, optimizer, loss, train_loader, valid_loader,
            target_transform, device, count_down_thr, max_epoch, tqdm_ncols)
        
        self._acc_k_top = acc_k_top

    def evaluate(self) -> Dict:
        self.model.eval()
        self.ram_cleanup()

        cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')        
        cross_entropy = torch.zeros(
            len(self.valid_loader),
            dtype=torch.float64,
            requires_grad=False).to(device=self.device)
        
        accuracy_top1 = torch.zeros(
            len(self.valid_loader),
            dtype=torch.float64,
            requires_grad=False).to(device=self.device)
        
        accuracy_topk = torch.zeros(
            len(self.valid_loader),
            dtype=torch.float64,
            requires_grad=False).to(device=self.device)

        with torch.no_grad():
            for idx, data in enumerate(tqdm(
                    self.valid_loader, total=len(self.valid_loader),
                    desc=" - S", ncols=self._tqdm_ncols)):
                data = data.to(self.device)
                
                outputs = self.model(data)
                targets = self.target_transform(data)
                cross_entropy[idx] = cross_entropy_loss(outputs, targets)
                
                _, top_k_idx = torch.topk(outputs, dim=1, k=self._acc_k_top)
                
                accuracy_top1[idx] = torch.sum(
                    torch.eq(top_k_idx[:, 0], targets)
                )/data.num_graphs
                
                accuracy_topk[idx] = torch.sum(torch.eq(
                    top_k_idx,
                    torch.broadcast_to(
                        targets.unsqueeze(dim=1),
                        (data.num_graphs, self._acc_k_top))
                ))/data.num_graphs

        cross_entropy = cross_entropy.to("cpu").numpy() 
        accuracy_top1 = accuracy_top1.to("cpu").numpy()
        accuracy_topk = accuracy_topk.to("cpu").numpy()
        
        self.ram_cleanup()
        return {
            "cross_entropy": (np.mean(cross_entropy), np.std(cross_entropy)),
            "accuracy_top1": (np.mean(accuracy_top1), np.std(accuracy_top1)),
            "accuracy_topk": (np.mean(accuracy_topk), np.std(accuracy_topk))
        }
    
    def print_progress(
            self, train_result, valid_result, evaluate_result,
            min_loss, counter, epoch) -> None:
        
        super().print_progress(
            train_result, valid_result, evaluate_result,
            min_loss, counter, epoch)
        
        print(
            f" - ACC(k=1): {evaluate_result['accuracy_top1'][0]:.3f}" +\
            f"; ACC(k={self._acc_k_top}): {evaluate_result['accuracy_top1'][0]:.3f}" +\
            f"; Counter: {counter:02d}"
        )
        return None
    
    def extract_result(
            self, train_result, valid_result, evaluate_result,
            min_loss, counter, epoch) -> Dict:
        return {
            "train_loss": train_result,
            "valid_loss": min_loss,
            "epoch": (epoch, 0),
            "k": (self._acc_k_top, 0),
            "accuracy_top1": evaluate_result["accuracy_top1"],
            "accuracy_topk": evaluate_result["accuracy_topk"],
            "cross_entropy": evaluate_result["cross_entropy"]
        }