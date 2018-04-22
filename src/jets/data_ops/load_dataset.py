import logging
import os
import pickle



def load_train_dataset(data_dir, filename, n_train, n_valid, do_preprocessing):
    if 'w-vs-qcd' in data_dir:
        from .w_vs_qcd import preprocess, crop_dataset
    elif 'quark-gluon' in data_dir:
        from .quark_gluon import preprocess, crop_dataset
    else:
        raise ValueError('Unrecognized data_dir!')
    #from problem_module import preprocess, crop_dataset

    problem = data_dir.split('/')[-1]
    subproblem = filename

    logging.warning("Loading data...")
    filename = "{}-train.pickle".format(filename)

    jets = load_jets(data_dir, filename, preprocess, preprocess_fn=preprocess)
    logging.warning("Found {} jets in total".format(len(jets)))

    if n_train > 0:
        jets = jets[:n_train]
    logging.warning("Splitting into train and validation...")
    #
    train_jets = jets[n_valid:]
    train_dataset = JetDataset(train_jets, problem=problem,subproblem=subproblem)
    #
    valid_jets = jets[:n_valid]
    valid_dataset = JetDataset(valid_jets, problem=problem,subproblem=subproblem)

    # crop validation set and add the excluded data to the training set
    valid_dataset, cropped_dataset = crop_dataset(valid_dataset)
    train_dataset.extend(cropped_dataset)

    train_dataset.shuffle()
    ##
    logging.warning("Building normalizing transform from training set...")
    train_dataset.transform()

    valid_dataset.transform(train_dataset.tf)

    # add cropped indices to training data
    logging.warning("\tfinal train size = %d" % len(train_dataset))
    logging.warning("\tfinal valid size = %d" % len(valid_dataset))

    return train_dataset, valid_dataset

def load_test_dataset(data_dir, filename, n_test, do_preprocessing):
    if 'w-vs-qcd' in data_dir:
        from .w_vs_qcd import preprocess, crop_dataset
    elif 'quark-gluon' in data_dir:
        from .quark_gluon import preprocess, crop_dataset
    else:
        raise ValueError('Unrecognized data_dir!')

    train_dataset, _ = load_train_dataset(data_dir, filename, -1, 27000, False)
    logging.warning("Loading test data...")
    filename = "{}-test.pickle".format(filename)
    jets = load_jets(data_dir, filename, preprocess)
    jets = jets[:n_test]

    dataset = JetDataset(jets)
    dataset.transform(train_dataset.tf)

    # crop validation set and add the excluded data to the training set
    dataset, _ = crop_dataset(dataset)

    # add cropped indices to training data
    logging.warning("\tfinal test size = %d" % len(dataset))

    return dataset
