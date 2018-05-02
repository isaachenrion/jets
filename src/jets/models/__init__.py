from .nmp import FixedNMP
from .recnn import RecursiveSimple, RecursiveGated
#from .recursive_net import RecursiveSimple
MODEL_DICT = dict(
    nmp=FixedNMP,
    recs=RecursiveSimple,
    recg=RecursiveGated
)
