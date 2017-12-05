
from .jet_transforms import TRANSFORMS, MESSAGE_PASSING_LAYERS, POOLINGS
from .predict import PredictFromParticleEmbedding


PREDICTORS = {
    'simple': (0, PredictFromParticleEmbedding)
}
