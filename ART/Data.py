from torch_geometric.data import Data

class GraphData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['graph_attr', 'y', 'y_cat'] or "mask" in key:
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)