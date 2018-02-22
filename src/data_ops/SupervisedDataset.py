from torch.utils.data import Dataset

class SupervisedDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def shuffle(self):
        perm = np.random.permutation(len(self.x))
        self.x = [self.x[i] for i in perm]
        self.y = [self.y[i] for i in perm]

    @classmethod
    def concatenate(cls, dataset1, dataset2):
        return cls(dataset1.x + dataset2.x, dataset1.y + dataset2.y)

    @property
    def dim(self):
        return self.x[0].size()[1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def extend(self, new_dataset):
        self.x = self.x + new_dataset.x
        self.y = self.y + new_dataset.y

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
