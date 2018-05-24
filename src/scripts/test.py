if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('Agg')
import sys
sys.path.append('../..')
'''
This is the top-level script for running test experiments on proteins or jets.

Usage:

python test.py p [argparse args] for proteins
python test.py j [argparse args] for jets


'''
if __name__ == "__main__":
    problem=sys.argv[1]
    if problem == 'p':
        from src.proteins.test.test_proteins import main
    elif problem == 'j':
        from src.jets.test.test_jets import main
    else:
        raise NotImplementedError("Testing scripts only implemented for jets and proteins")
    main(sys.argv[2:])
