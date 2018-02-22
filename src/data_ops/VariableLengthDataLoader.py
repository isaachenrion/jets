import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

def wrap(x):
    x = Variable(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

class VariableLengthDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, return_mask=False):
        self.return_mask = return_mask
        super().__init__(dataset, batch_size, collate_fn=self.collate)

    def collate(self, xy_pairs):
        data = [x for x, y in xy_pairs]
        y = torch.stack([y for x, y in xy_pairs], 0)
        if y.size()[1] == 1:
            y = y.squeeze(1)
        y = wrap(y)

        seq_lengths = [len(x) for x in data]
        max_seq_length = max(seq_lengths)
        padded_data = torch.zeros(len(data), max_seq_length, self.dataset.dim)
        for i, x in enumerate(data):
            padded_data[i][:len(x)] = x
        padded_data = wrap(padded_data)

        if self.return_mask:
            mask = torch.ones(len(data), max_seq_length, max_seq_length)
            for i, x in enumerate(data):
                seq_length = len(x)
                if seq_length < max_seq_length:
                    mask[i, seq_length:, :].fill_(0)
                    mask[i, :, seq_length:].fill_(0)
            mask = wrap(mask)

            return padded_data, y, mask
        return padded_data, y
