from typing import AbstractSet, Any, NamedTuple, Tuple, List, Union, OrderedDict


class LayerParSet():

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
            raise RuntimeError
        return getattr(self, key)

    def __repr__(self) -> str:
        attrs = [None] * len(self.keys)
        
        for i, key in enumerate(self.keys):
            attrs[i] = f"{key}={getattr(self, key)}"
        attrs_str = ", ".join(attrs)
        return f"LayerParSet({attrs_str})"


class MultiLayerParSet():

    def __init__(
            self,
            in_channels: int = 64,
            hidden_channels : Union[List[int], None] = None,
            out_channels: int = 1,
            **kwargs
        ) -> None:
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.other_fields = {}
        for key, value in kwargs.items():
            self.other_fields[key] = value

    def unwind(self) -> OrderedDict[int, LayerParSet]:
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
            
            lyr_pars[i] = LayerParSet(
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

class AttentiveFPPars(NamedTuple):
    in_channels: int
    hidden_channels: int
    out_channels: int
    edge_dim: int
    num_layers: int
    num_timesteps: int
    dropout: float = 0.0
