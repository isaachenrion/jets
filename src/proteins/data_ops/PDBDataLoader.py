import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data_ops.pad_tensors import pad_tensors_extra_channel, pad_tensors, pad_matrices

def collate_protein_tuples(protein_tuples):
    sequences, coords = list(map(list, zip(*protein_tuples)))
    sequences, batch_mask = pad_tensors_extra_channel(sequences)
    coords, _ = pad_tensors(coords)

    coords_mask = torch.ones(coords.shape[0], coords.shape[1], coords.shape[1])
    bs_idx, v_idx = np.where(coords[:,:,0] == np.inf)
    coords_mask[bs_idx, v_idx, :] = 0
    coords_mask[bs_idx, :, v_idx] = 0
    coords_mask = coords_mask * batch_mask

    batch = (sequences, coords, batch_mask, coords_mask)
    return batch

class PDBDataLoader(DataLoader):
    def __init__(self, pdb_dataset, batch_size):
        super().__init__(dataset=pdb_dataset, batch_size=batch_size, collate_fn=collate_protein_tuples)

    @property
    def xdim(self):
        return self.dataset.xdim+1
