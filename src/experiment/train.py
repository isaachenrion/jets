
def train(**kwargs):
    dataset = kwargs['data_args'].dataset
    if dataset in ['w', 'pp', 'pbpb']:
        from src.jets.Training import Training
    elif dataset in ['protein']:
        from src.proteins.Training import Training

    Training(**kwargs) 
