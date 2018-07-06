from .model_loading import build_model, load_model
from .generic_train_script import generic_train_script
from .generic_test_script import generic_test_script

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
