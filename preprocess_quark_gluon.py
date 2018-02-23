from src.data_ops.preprocessing.quark_gluon import mix_and_pickle

if __name__ == '__main__':
    data_dir = 'data/quark-gluon/preprocessed'
    env = 'pp'
    mix_and_pickle(data_dir, env)
