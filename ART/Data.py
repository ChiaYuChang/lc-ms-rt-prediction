from torch_geometric.data import Data

class GraphData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        # if key in ['graph_attr', 'y', 'y_cat', 'mw_ppm'] or "mask" in key:
        if key in ['y', 'y_cat', 'mw_ppm', 'ratio'] or "mask" in key:
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)
    
    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return self.num_nodes
        elif 'tt' in key and "batch" in key:
            if 'graph' in key:
                return 1
            elif 'node' in key:
                return self.num_tts
            else:
                return 0
        else:
            return 0
