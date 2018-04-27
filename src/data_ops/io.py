import pickle

def save_objects_to_pickle(objs, filename):
    obj_dicts = [vars(obj) for obj in objs]
    save_dicts_to_pickle(obj_dicts, filename)

def save_dicts_to_pickle(jet_dicts, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dicts, f)

def load_dicts_from_pickle(filename):
    with open(filename, 'rb') as f:
        dicts = pickle.load(f, encoding='latin-1')
    return dicts
