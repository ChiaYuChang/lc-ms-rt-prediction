from typing import Any, NamedTuple, Tuple, List, Union, OrderedDict


class LayerParSet():
    __name__ = "LayerParSet"

    def __init__(
            self,
            in_channels: int,
            out_channels: int
            ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels


class LinearLayerPars(LayerParSet):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout: Tuple[float, bool] = (0.1, False), # (p, inplace)
            relu: bool = False,
            batch_norm: bool = False
            ) -> None:
        super().__init__(in_channels, out_channels)
        self.dropout = dropout
        self.relu = relu
        self.batch_norm = batch_norm


class MultiLayerParSet():

    def __init__(
            self,
            in_channels: int = 64,
            hidden_channels : Union[List[int], None] = None,
            out_channels: int = 1
        ) -> None:
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

    def unwind(self) -> OrderedDict[int, LayerParSet]:
        raise NotImplementedError
    
    def broadcast(self, p, l):
        if not(isinstance(p, List)):
            p = [p]
        if len(p) == l:
            return p
        else:
            q = p * (l//len(p))
            r = l % len(p)
            if not(r == 0):
                print("longer object length is not a multiple of shorter object length")
                q += p[0:r]
            return q


class PredictorPars(MultiLayerParSet):
    
    def __init__(
            self,
            in_channels: int = 64,
            hidden_channels: Union[List[int], None] = None,
            out_channels: int = 1,
            dropout: Union[List[Tuple[float, bool]], Tuple[float, bool]] = 0.1,
            relu: Union[List[bool], bool] = False,
            batch_norm: Union[List[bool], bool] = False
            ) -> None:
        super().__init__(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)
        self.dropout = dropout
        self.relu = relu
        self.batch_norm = batch_norm
    
    def unwind(self) -> OrderedDict[int, LayerParSet]:
        in_channels = [self.in_channels] + self.hidden_channels
        out_channels = self.hidden_channels + [self.out_channels]
        n_layer = len(in_channels)
        dropout = self.broadcast(self.dropout, n_layer)
        relu = self.broadcast(self.relu, n_layer)
        batch_norm = self.broadcast(self.batch_norm, n_layer)
        
        lyr_pars = OrderedDict()
        for i in range(n_layer):
            lyr_pars[i] = LinearLayerPars(
                in_channels = in_channels[i],
                out_channels = out_channels[i],
                dropout = dropout[i],
                relu = relu[i],
                batch_norm = batch_norm[i]
            )
        return lyr_pars

class AttentiveFPPars(NamedTuple):
    in_channels: int
    hidden_channels: int
    out_channels: int
    edge_dim: int
    num_layers: int
    num_timesteps: int
    dropout: float = 0.0
