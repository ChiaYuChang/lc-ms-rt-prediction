import torch
import torch.nn as nn

from torch.nn.init import xavier_normal_ as glorot_
from typing import Tuple

class GraphConvLayer(nn.Module):
    def __init__(
            self,
            in_channels: int = 128,
            out_channels: int = 128,
            dropout: Tuple[float, bool] = 0.0,
            relu: bool = True,
            batch_norm: bool = False,
            **kwargs):
        super().__init__(**kwargs)

        self.dropout = nn.Dropout(p=dropout[0], inplace=dropout[1])

        self.W_0 = nn.parameter.Parameter(
            torch.empty((in_channels, out_channels), dtype=torch.float32)
        )
        self.W_1 = nn.parameter.Parameter(
            torch.empty((in_channels, out_channels), dtype=torch.float32)
        )
        # self.activation = nn.ReLU()
        # self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
        if relu:
            self.activation = nn.ReLU()
        else:
            self.activation = lambda x: x

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
        else:
            self.batch_norm = lambda x: x


        # self.supports_masking = True

    # def compute_mask(self, inputs, mask=None):
    #     if mask is None:
    #         return None
    #     return mask[1]

    def reset_parameters(self):
        glorot_(self.W_0)
        glorot_(self.W_1)

    def forward(self, tilde_A, node_attr):
        # \tilde{A} = D^{-1/2}AD^{-1/2}
        A = tilde_A
        H0 = node_attr

        H1 = torch.add(torch.matmul(H0, self.W_0), torch.linalg.multi_dot([A, H0, self.W_1]))
        H1 = self.batch_norm(H1)
        H1 = self.activation(H1)
        H1 = self.dropout(H1)

        # if mask:
        #     H_mask = mask[1][:, :, None]
        #     H *= tf.cast(H_mask, H.dtype)
        return H1
