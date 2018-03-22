

def load_train_dataset(data_dir, *args, **kwargs):
    if 'w-vs-qcd' in data_dir:
        from .jets.load_dataset import load_train_dataset as ltd
    elif 'quark-gluon' in data_dir:
        from .jets.load_dataset import load_train_dataset as ltd
    elif 'protein' in data_dir:
        from .proteins.load_dataset import load_train_dataset as ltd
    return ltd(data_dir, *args, **kwargs)


def load_test_dataset(data_dir, *args, **kwargs):
    if 'w-vs-qcd' in data_dir:
        from .jets.load_dataset import load_test_dataset as ltd
    elif 'quark-gluon' in data_dir:
        from .jets.load_dataset import load_test_dataset as ltd
    elif 'protein' in data_dir:
        from .proteins.load_dataset import load_test_dataset as ltd
    return ltd(data_dir, *args, **kwargs)
