from ..utils import _DataOps
from .TreeDataset import TreeDataset
from .TreeDataLoader import TreeDataLoader
from .TreeJet import TreeJet

class TreeDataOps(_DataOps):
    DataLoader = TreeDataLoader
    Dataset = TreeDataset
    Jet = TreeJet

    def __init__(self):
        super().__init__()
