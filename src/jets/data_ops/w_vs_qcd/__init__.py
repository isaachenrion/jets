from .leaf import LeafDataOps
from .tree import TreeDataOps

def get_train_data_loader(leaves=None,**kwargs):
    if leaves is None:
        raise ValueError
    if leaves:
        return LeafDataOps.get_train_data_loader(**kwargs)
    return TreeDataOps.get_train_data_loader(**kwargs)

def get_test_data_loader(leaves=None,**kwargs):
    if leaves is None:
        raise ValueError
    if leaves:
        return LeafDataOps.get_test_data_loader(**kwargs)
    return TreeDataOps.get_test_data_loader(**kwargs)
