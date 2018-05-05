import os
import logging

from ._load_jets import _load_jets
from .preprocessing import preprocess
from . import utils

class _DataOps:
    Jet = lambda: 1/0
    DataLoader = lambda: 1/0
    Dataset = lambda: 1/0
    preprocess_fn = preprocess
    crop_pileup_jets = utils.crop_pileup_jets
    crop_original_jets = utils.crop_original_jets
    flatten_pileup_jets = utils.flatten_pileup_jets
    flatten_original_jets = utils.flatten_original_jets

    w_vs_qcd = 'w-vs-qcd'
    quark_gluon = 'quark-gluon'
    DATASETS = {
        'w':(w_vs_qcd,'antikt-kt'),
        'wp':(w_vs_qcd + '/pileup','pileup'),
        'pp': (quark_gluon,'pp'),
        'pbpb': (quark_gluon,'pbpb'),
        'wd':(w_vs_qcd, 'medium')
    }

    @classmethod
    def load_jets(cls, data_dir, filename, do_preprocessing=False):
        jets = _load_jets(data_dir, filename, cls.Jet, cls.preprocess_fn, do_preprocessing)
        return jets

    @classmethod
    def training_and_validation_dataset(cls, data_dir, dataset, n_train, n_valid, preprocess):

        intermediate_dir, filename = cls.DATASETS.get(dataset, None)
        data_dir = os.path.join(data_dir, intermediate_dir)
        jets = cls.load_jets(data_dir,"{}-train.pickle".format(filename), preprocess)

        problem = data_dir.split('/')[-1]
        subproblem = filename

        pileup = 'pileup' in filename
        if pileup:
            crop = cls.crop_pileup_jets
            flatten_jets = cls.flatten_pileup_jets
        else:
            crop = cls.crop_original_jets
            flatten_jets = cls.flatten_original_jets

        good_jets, bad_jets = crop(jets)

        valid_jets = good_jets[:n_valid]
        train_jets = (good_jets[n_valid:] + bad_jets)

        if n_train >= 0:
            train_jets = train_jets[:n_train]
        dummy_train_jets = good_jets[n_valid:2*n_valid]

        train_dataset = cls.Dataset(train_jets)
        valid_dataset = cls.Dataset(valid_jets, weights=flatten_jets(valid_jets))
        dummy_train_dataset = cls.Dataset(dummy_train_jets, weights=flatten_jets(dummy_train_jets))

        train_dataset.shuffle()

        ##
        #logging.warning("Building normalizing transform from training set...")
        #train_dataset.transform()
        #valid_dataset.transform(train_dataset.tf)
        #dummy_train_dataset.transform(train_dataset.tf)

        # add cropped indices to training data
        logging.warning("\tfinal train size = %d" % len(train_dataset))
        logging.warning("\tfinal valid size = %d" % len(valid_dataset))
        logging.warning("\tfinal dummy train size = %d" % len(dummy_train_dataset))

        return train_dataset, valid_dataset, dummy_train_dataset

    @classmethod
    def test_dataset(cls, data_dir, dataset, n_test, preprocess):
        train_dataset, _, _ = cls.training_and_validation_dataset(data_dir, dataset, -1, 27000, False)

        intermediate_dir, filename = cls.DATASETS.get(dataset, None)
        data_dir = os.path.join(data_dir, intermediate_dir)

        logging.warning("Loading test data...")
        filename = "{}-test.pickle".format(filename)
        jets = cls.load_jets(data_dir, filename, preprocess)

        pileup = 'pileup' in filename
        if pileup:
            crop = cls.crop_pileup_jets
            flatten_jets = cls.flatten_pileup_jets
        else:
            crop = cls.crop_original_jets
            flatten_jets = cls.flatten_original_jets


        good_jets, bad_jets = crop(jets)
        test_jets = good_jets[:n_test]
        dataset = cls.Dataset(test_jets, weights=flatten_jets(test_jets))

        #dataset.transform(train_dataset.tf)
        # add cropped indices to training data
        logging.warning("\tfinal test size = %d" % len(dataset))

        return dataset

    @classmethod
    def get_train_data_loader(cls, data_dir, dataset, n_train, n_valid, batch_size, preprocess=None,**kwargs):
        train_dataset, valid_dataset, dummy_train_dataset = cls.training_and_validation_dataset(data_dir, dataset, n_train, n_valid, preprocess)
        train_data_loader = cls.DataLoader(train_dataset, batch_size, **kwargs)
        valid_data_loader = cls.DataLoader(valid_dataset, batch_size, **kwargs)
        dummy_train_data_loader = cls.DataLoader(dummy_train_dataset, batch_size,**kwargs)

        return train_data_loader, valid_data_loader, dummy_train_data_loader

    @classmethod
    def get_test_data_loader(cls, data_dir, dataset, n_test, batch_size, preprocess=None,**kwargs):
        dataset = cls.test_dataset(data_dir, dataset, n_test, preprocess)
        test_data_loader = cls.DataLoader(dataset, batch_size, **kwargs)
        return test_data_loader
