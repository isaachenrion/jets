import os
import logging
import pickle

import numpy as np
from sklearn.preprocessing import RobustScaler

from ._load_jets import _load_jet_dicts
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
        #'pp': (quark_gluon,'pp'),
        #'pbpb': (quark_gluon,'pbpb'),
        'wd':(w_vs_qcd, 'medium'),
        'desc': (w_vs_qcd, 'antikt-seqpt-reversed')
    }

    @classmethod
    def load_jet_dicts(cls, dataset_type, data_dir, dataset, do_preprocessing=False):
        intermediate_dir, filename = cls.DATASETS.get(dataset, None)
        data_dir = os.path.join(data_dir, intermediate_dir)
        if dataset_type == 'train':
            suffix = "-train.pickle"
        elif dataset_type == 'test':
            suffix = "-test.pickle"
        else:
            raise ValueError
        jet_dicts = _load_jet_dicts(data_dir, filename + suffix, cls.preprocess_fn, do_preprocessing)
        return jet_dicts

    @classmethod
    def load_jets(cls, jet_dicts):
        jets = [cls.Jet(**jd) for jd in jet_dicts]
        logging.warning("\tSuccessfully loaded data")
        logging.warning("\tFound {} jets in total".format(len(jets)))
        return jets

    @classmethod
    def crop_and_flatten(cls, jets, pileup):
        #pileup = 'pileup' in filename
        if pileup:
            crop = cls.crop_pileup_jets
            flatten_jets = cls.flatten_pileup_jets
        else:
            crop = cls.crop_original_jets
            flatten_jets = cls.flatten_original_jets

        good_jets, bad_jets = crop(jets)
        good_weights = flatten_jets(good_jets)
        #bad_weights = flatten_jets(bad_jets)

        return good_jets, good_weights, bad_jets

    @classmethod
    def get_transform(cls, data_dir, dataset):
        intermediate_dir, _ = cls.DATASETS.get(dataset, None)
        complete_data_dir = os.path.join(data_dir, intermediate_dir)
        transform_dir = os.path.join(complete_data_dir, 'transform')
        transform_filename = os.path.join(transform_dir, dataset)

        if os.path.exists(transform_filename):
            logging.info("Loading transform")
            with open(transform_filename, 'rb') as f:
                transform = pickle.load(f, encoding='latin-1')
            return transform
        else:
            logging.info("Building transform")
            if not os.path.exists(transform_dir):
                os.makedirs(transform_dir)
            jet_dicts = cls.load_jet_dicts('train', data_dir, dataset, False)
            transform = RobustScaler().fit(np.vstack([jd["tree_content"] for jd in jet_dicts]))
            with open(transform_filename, 'wb') as f:
                pickle.dump(transform, f)
            return transform


    @classmethod
    def training_and_validation_dataset(cls, data_dir, dataset, n_train, n_valid, do_preprocessing):

        jet_dicts = cls.load_jet_dicts('train', data_dir, dataset, do_preprocessing)
        tf = cls.get_transform(data_dir, dataset)
        for jet_dict in jet_dicts:
            jet_dict["tree_content"] = tf.transform(jet_dict["tree_content"])
        jets = cls.load_jets(jet_dicts)

        valid_jets = jets[:n_valid]
        train_jets = jets[n_valid:]
        if n_train >= 0:
            train_jets = train_jets[:n_train]

        good_valid_jets, good_valid_weights, bad_valid_jets = cls.crop_and_flatten(valid_jets, dataset == 'wp')
        valid_jets = good_valid_jets
        dummy_train_jets = bad_valid_jets + good_valid_jets
        valid_weights = good_valid_weights
        dummy_train_weights = None

        train_dataset = cls.Dataset(train_jets)
        valid_dataset = cls.Dataset(valid_jets, weights=valid_weights)
        dummy_train_dataset = cls.Dataset(dummy_train_jets, weights=dummy_train_weights)

        train_dataset.shuffle()

        logging.warning("\tfinal train size = %d" % len(train_dataset))
        logging.warning("\tfinal valid size = %d" % len(valid_dataset))
        logging.warning("\tfinal dummy train size = %d" % len(dummy_train_dataset))

        return train_dataset, valid_dataset, dummy_train_dataset

    @classmethod
    def test_dataset(cls, data_dir, dataset, n_test, do_preprocessing):
        jet_dicts = cls.load_jet_dicts('test', data_dir, dataset, do_preprocessing)
        tf = cls.get_transform(data_dir, dataset)
        for jet_dict in jet_dicts:
            jet_dict["tree_content"] = tf.transform(jet_dict["tree_content"])
        jets = cls.load_jets(jet_dicts)

        good_jets, good_weights, _ = cls.crop_and_flatten(jets, dataset == 'wp')

        test_jets = good_jets[:n_test] if n_test > 0 else good_jets
        test_weights = good_weights[:n_test] if n_test > 0 else good_weights
        dataset = cls.Dataset(test_jets, weights=test_weights)

        logging.warning("\tfinal test size = %d" % len(dataset))

        return dataset

    @classmethod
    def get_train_data_loader(cls, data_dir, dataset, n_train, n_valid, batch_size, do_preprocessing,**kwargs):
        train_dataset, valid_dataset, dummy_train_dataset = cls.training_and_validation_dataset(data_dir, dataset, n_train, n_valid, do_preprocessing)
        train_data_loader = cls.DataLoader(train_dataset, batch_size, **kwargs)
        valid_data_loader = cls.DataLoader(valid_dataset, batch_size, **kwargs)
        dummy_train_data_loader = cls.DataLoader(dummy_train_dataset, batch_size,**kwargs)

        return train_data_loader, valid_data_loader, dummy_train_data_loader

    @classmethod
    def get_test_data_loader(cls, data_dir, dataset, n_test, batch_size, do_preprocessing, **kwargs):
        dataset = cls.test_dataset(data_dir, dataset, n_test, do_preprocessing)
        test_data_loader = cls.DataLoader(dataset, batch_size, **kwargs)
        return test_data_loader
