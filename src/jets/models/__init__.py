from .nmp import FixedNMP
from .recursive_net import RecursiveSimple
MODEL_DICT = dict(
    nmp=FixedNMP,
    recs=RecursiveSimple
)
