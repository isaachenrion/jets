if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('Agg')
import sys
sys.path.append('../..')
if __name__ == "__main__":
    problem=sys.argv[1]
    if problem == 'p':
        from src.proteins.train import train
    else:
        raise NotImplementedError
    train(sys.argv[2:])
