
#from .jet_transforms import TRANSFORMS, MESSAGE_PASSING_LAYERS, POOLINGS, ADAPTIVE_MATRICES
#from .jet_transforms import construct_transform
from .predict import PredictFromParticleEmbedding


#PREDICTORS = {
#    'simple': (0, PredictFromParticleEmbedding)
#}
from misc.abstract_constructor import construct_object

def construct_classifier(key, *args, **kwargs):
    dictionary = dict(
        simple=PredictFromParticleEmbedding
    )
    try:
        return construct_object(key, dictionary, *args, **kwargs)
    except ValueError as e:
        raise ValueError('Classifier {}'.format(e))
