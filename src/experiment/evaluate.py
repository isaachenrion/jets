
def evaluate(**kwargs):
    dataset = kwargs['data_args'].dataset
    if dataset in ['w', 'pp', 'pbpb']:
        from src.jets.Evaluation import Evaluation
    elif dataset in ['protein']:
        from src.proteins.Evaluation import Evaluation

    Evaluation(**kwargs)
