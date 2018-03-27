from torch.utils.data import DataLoader

class _DataLoader(DataLoader):
    def __init__(self, dataset, batch_size):
        super().__init__(dataset, batch_size, collate_fn=self.collate)

    def collate(self, xy_pairs):
        X = self.preprocess_x([x for x, _ in xy_pairs])
        Y = self.preprocess_y([y for _, y in xy_pairs])
        return X, Y

    def preprocess_x(self, x_list):
        pass

    def preprocess_y(self, y_list):
        pass
