import torch
from torch.utils.data import Dataset
class PDBDataset(Dataset):
    def __init__(self, sequences, coords):
        super().__init__()
        try:
            assert len(sequences) == len(coords)
        except AssertionError:
            raise ValueError
        self.sequences = sequences
        self.coords = coords

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.coords[idx])

    def __len__(self):
        return len(self.sequences)

    @property
    def xdim(self):
        return self.sequences[0].shape[1]
