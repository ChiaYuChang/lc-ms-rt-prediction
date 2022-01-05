# %%
from ART.Deprecated.SMRT import SMRT
from torch_geometric.loader import DataLoader

# %%
smrt = SMRT(
    root="./SMRT",
    profile_name="SMRT.json",
    split="train",
    max_num_tautomer=5
)
# %%
smrt_loader = DataLoader(smrt, batch_size=100)

# %%
