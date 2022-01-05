import torch

from collections import OrderedDict
from torch import nn
from torch.nn.init import xavier_normal_ as glorot_
from torch_geometric.nn import BatchNorm
from torch_geometric.nn.models import AttentiveFP
from typing import Dict, Union, List

from .ParSet import LinearLayerPars, AttentiveFPPars, PredictorPars


class ARTNet(nn.Module):
    def __init__(
            self,
            embd_lyr_pars_dict: Dict[str, LinearLayerPars],
            afp_mdl_pars: AttentiveFPPars,
            prdctr_lyr_pars: PredictorPars
            ) -> None:
        super(ARTNet, self).__init__()

        self.embdng_lyrs = nn.ModuleDict({})
        for key in embd_lyr_pars_dict.keys():
            nn_mdl = nn.Sequential()
            nn_mdl.add_module(
                name="lnr_{}".format(key),
                module=nn.Linear(
                        in_features=embd_lyr_pars_dict[key].in_features,
                        out_features=embd_lyr_pars_dict[key].out_features
                )
            )
            if embd_lyr_pars_dict[key].dropout[0] > 0:
                nn_mdl.add_module(
                    name="drp_{}".format(key),
                    module=nn.Dropout(
                        p=embd_lyr_pars_dict[key].dropout[0],
                        inplace=embd_lyr_pars_dict[key].dropout[1]
                    )
                )
            if embd_lyr_pars_dict[key].relu is True:
                nn_mdl.add_module(
                    name="relu_{}".format(key),
                    module=nn.ReLU()
                )

            if embd_lyr_pars_dict[key].batch_norm is True:
                nn_mdl.add_module(
                    name="btch_nrm_{}".format(key),
                    module=nn.BatchNorm1d(
                        num_features=embd_lyr_pars_dict[key].out_features
                    )
                )
            self.embdng_lyrs[key] = nn_mdl

        self.afp_mdl = AttentiveFP(
            in_channels=afp_mdl_pars.in_channels,
            hidden_channels=afp_mdl_pars.hidden_channels,
            out_channels=afp_mdl_pars.out_channels,
            edge_dim=afp_mdl_pars.edge_dim,
            num_layers=afp_mdl_pars.num_layers,
            num_timesteps=afp_mdl_pars.num_timesteps,
            dropout=afp_mdl_pars.dropout
        )

        self.afp_btch_nrm = BatchNorm(
            in_channels=afp_mdl_pars.out_channels
            + embd_lyr_pars_dict["mol_attr"].out_features
            + 3
        )

        prdctr_lyr_pars = self._predictor_par_transform(
            in_features=prdctr_lyr_pars.in_features,
            hidden_features=prdctr_lyr_pars.hidden_features,
            out_features=prdctr_lyr_pars.out_features,
            dropout=prdctr_lyr_pars.dropout,
            relu=prdctr_lyr_pars.relu,
            batch_norm=prdctr_lyr_pars.batch_norm
        )

        self.prdctr = nn.Sequential()
        for i in prdctr_lyr_pars.keys():
            self.prdctr.add_module(
                name="lnr_{}".format(i),
                module=nn.Linear(
                    in_features=prdctr_lyr_pars[i].in_features,
                    out_features=prdctr_lyr_pars[i].out_features
                )
            )
            if prdctr_lyr_pars[i].dropout[0] > 0:
                self.prdctr.add_module(
                    name="drp_{}".format(i),
                    module=nn.Dropout(
                        p=prdctr_lyr_pars[i].dropout[0],
                        inplace=prdctr_lyr_pars[i].dropout[1]
                    )
                )
            if prdctr_lyr_pars[i].relu is True:
                self.prdctr.add_module(
                    name="relu_{}".format(i),
                    module=nn.ReLU()
                )
        self.reset_parameters()

    def reset_parameters(self):
        def init_weights_(m, bias=0.01):
            if isinstance(m, nn.Linear):
                glorot_(m.weight)
                m.bias.data.fill_(bias)

        for lyr_name in self.embdng_lyrs.keys():
            self.embdng_lyrs[lyr_name].apply(init_weights_)

        self.afp_mdl.reset_parameters()

        self.prdctr.apply(init_weights_)

    def _predictor_par_transform(
            self,
            in_features: int = 64,
            hidden_features: Union[List[int], None] = None,
            out_features: int = 1,
            dropout: Union[List[float], float] = 0.1,
            relu: Union[List[bool], bool] = False,
            batch_norm: Union[List[bool], bool] = False
            ) -> OrderedDict:

        in_features = [in_features] + hidden_features
        out_features = hidden_features + [out_features]
        n_layer = len(in_features)

        if isinstance(dropout, float):
            dropout = [dropout] * n_layer
            dropout[-1] = -1
        if isinstance(relu, bool):
            relu = [relu] * n_layer
            relu[-1] = False
        if isinstance(batch_norm, bool):
            batch_norm = [batch_norm] * n_layer
            batch_norm[-1] = False

        prdctr_lyr_pars = OrderedDict()
        for i in range(n_layer):
            prdctr_lyr_pars[i] = LinearLayerPars(
                    in_features=in_features[i],
                    out_features=out_features[i],
                    dropout=(dropout[i], True),
                    relu=relu[i],
                    batch_norm=batch_norm[i]
            )
        return prdctr_lyr_pars

    def forward(self, data):
        # embedding categorical atom attributes:
        # x = embdng_lyrs["node_attrs"](data.x).view(data.num_nodes, -1)
        embd_node_attr = self.embdng_lyrs["node_attr"](data.x)
        embd_edge_attr = self.embdng_lyrs["edge_attr"](data.edge_attr)
        afp = self.afp_mdl(
            x=embd_node_attr,
            edge_index=data.edge_index,
            edge_attr=embd_edge_attr,
            batch=data.batch
        )

        embd_mol_dscrt_attr = self.embdng_lyrs["mol_attr"](
            data.mol_attr.view((len(data.SMILES), -1))
        )

        embd_mol_cntns_attr = torch.stack((
            data.mlogP, data.volumn, data.wt
        ), dim=1)

        afp = torch.cat((afp, embd_mol_dscrt_attr, embd_mol_cntns_attr), dim=1)
        afp = self.afp_btch_nrm(afp)
        y = self.prdctr(afp)
        return y