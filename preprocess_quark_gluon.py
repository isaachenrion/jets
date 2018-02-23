from src.data_ops.quark_gluon import mix_and_pickle

if __name__ == '__main__':
    data_dir = 'data/quark-gluon'
    env = 'pbpb'
    mix_and_pickle(data_dir, env)
