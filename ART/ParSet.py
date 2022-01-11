from typing import AbstractSet, Any, Callable, NamedTuple, Optional, Tuple, List, Type, Union, OrderedDict
import torch
import torch_geometric


class LayerParSet():
    __name__ = "LayerParSet"
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            **kwargs
            ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._keys = ["in_channels", "out_channels"]
        for key, value in kwargs.items():
            self._keys += [key]
            setattr(self, key, value)
    
    @property
    def keys(self):
        return self._keys

    def __getattr__(self, key: str) -> Any:
        if key not in self._keys:
            raise AttributeError(f"The attribute '{key}' could not be found.")
        return getattr(self, key)

    def __repr__(self) -> str:
        attrs = [None] * len(self.keys)
        
        for i, key in enumerate(self.keys):
            attrs[i] = f"{key}={getattr(self, key)}"
        attrs_str = ", ".join(attrs)
        return f"{self.__name__}({attrs_str})"


class LinearLayerParSet(LayerParSet):
    __name__ = "LinearLayerParSet"
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool,
            dropout: Tuple[float, bool],
            relu: True,
            batch_norm=True
            ) -> None:
        super().__init__(
            in_channels=in_channels, 
            out_channels=out_channels,
            bias=bias,
            dropout=dropout,
            relu=relu,
            batch_norm=batch_norm)


class GCNLayerParSet(LayerParSet):
    __name__ = "GCNLayerParSet"
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout: Tuple[float, bool],
            relu: True,
            batch_norm=True) -> None:
        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            dropout=dropout,
            relu=relu,
            batch_norm=batch_norm)


class GATConvLayerPar(LayerParSet):
    __name__ = "GATConvLayerPar"
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads:int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: Tuple[float, bool] = (0.0, False),
            add_self_loops: bool = True,
            edge_dim: Optional[int] = None,
            bias: bool = True,
             **kwargs) -> None:
        
        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            heads = heads,
            concat = concat,
            negative_slope = negative_slope,
            dropout = dropout,
            add_self_loops = add_self_loops,
            edge_dim = edge_dim,
            bias = bias,
            **kwargs)


class SAGPoolingPar(LayerParSet):
    __name__ = "SAGPoolingPar"
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            ratio: Union[float, int] = 0.5,
            GNN: Callable = torch_geometric.nn.conv.GraphConv,
            min_score: Optional[float] = None,
            multiplier: Optional[float] = 1,
            nonlinearity: Callable = torch.tanh,
            **kwargs) -> None:
        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            ratio = ratio,
            GNN = GNN,
            min_score = min_score,
            multiplier = multiplier,
            nonlinearity = nonlinearity,
            **kwargs)


class MultiLayerParSet():
    __name__ = "MultiLayerParSet"
    
    def __init__(
            self,
            in_channels: int = 64,
            hidden_channels : Union[List[int], None] = None,
            out_channels: int = 1,
            output_obj: Optional[Type] = LayerParSet,
            **kwargs
        ) -> None:
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        if not(issubclass(output_obj, LayerParSet)):
            raise TypeError("OutputObj should be a subclass of LayerParSet")
        self.output_obj = output_obj

        self.other_fields = {}
        for key, value in kwargs.items():
            self.other_fields[key] = value

    def unwind(self) -> List[LayerParSet]:

        in_channels = [self.in_channels] + self.hidden_channels
        out_channels = self.hidden_channels + [self.out_channels]
        n_layer = len(in_channels)
        
        for key in self.other_fields.keys():
            self.other_fields[key] = self.broadcast(self.other_fields[key], n_layer)
    
        lyr_pars = [None] * n_layer
        for i in range(n_layer):
            kwargs_dict = {}
            
            for key in self.other_fields.keys():
                kwargs_dict[key] = self.other_fields[key][i]
            
            lyr_pars[i] = self.output_obj(
                in_channels = in_channels[i],
                out_channels = out_channels[i],
                **kwargs_dict
            )
        return lyr_pars

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


LayerParSetType = Union[MultiLayerParSet, List[LayerParSet], LayerParSet]