from ..utils import _DataOps
from .LeafJet import LeafJet
from .LeafDataLoader import LeafDataLoader
from .LeafDataset import LeafDataset


class LeafDataOps(_DataOps):
    DataLoader = LeafDataLoader
    Dataset = LeafDataset
    Jet = LeafJet
