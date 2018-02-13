from .predict import TreeJetClassifier
from .predict import LeafJetClassifier

def construct_classifier(key, *args, **kwargs):
    jet_transform = kwargs.get('jet_transform', None)
    assert jet_transform is not None
    if jet_transform in ['recs', 'recg']:
        return TreeJetClassifier(**kwargs)
    else:
        return LeafJetClassifier(**kwargs)
    
