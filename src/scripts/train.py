if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('Agg')
import sys
sys.path.append('../..')
if __name__ == "__main__":
    problem=sys.argv[1]
    if problem == 'p':
        from src.proteins.train import main
    elif problem == 'j':
        from src.jets.train import main
    else:
        raise NotImplementedError("Training scripts only implemented for jets and proteins")
    main(sys.argv[2:])
