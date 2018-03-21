

def load_train_dataset(data_dir, filename, n_train, n_valid, redo):
    if 'w-vs-qcd' in data_dir:
        from .jets.load_dataset import load_train_dataset as ltd
    elif 'quark-gluon' in data_dir:
        from .jets.load_dataset import load_train_dataset as ltd
    elif 'protein' in data_dir:
        from .proteins.load_dataset import load_train_dataset as ltd
    return ltd(data_dir, filename, n_train, n_valid, redo)


def load_test_dataset(data_dir, filename, n_test, redo):
    if 'w-vs-qcd' in data_dir:
        from .jets.load_dataset import load_test_dataset as ltd
    elif 'quark-gluon' in data_dir:
        from .jets.load_dataset import load_test_dataset as ltd
    elif 'protein' in data_dir:
        from .proteins.load_dataset import load_test_dataset as ltd
    return ltd(data_dir, filename, n_test, redo)
