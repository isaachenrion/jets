from torch.utils.data import Dataset

class JetDataset(Dataset):
    def __init__(self, jets):
        super().__init__()
        self.jets = jets

    def __len__(self):
        return len(self.jets)

    def __getitem__(self, idx):
        return self.jets[idx], self.jets[idx].y

    @property
    def dim(self):
        return self.jets[0].constituents.shape[1]

    def extend(self, dataset):
        self.jets = self.jets + dataset.jets

    @classmethod
    def concatenate(cls, dataset1, dataset2):
        return cls(dataset1.jet + dataset2.jets)


class WeightedJetDataset(JetDataset):
    def __init__(self, jets, w):
        super().__init__(jets)
        self.w = w
