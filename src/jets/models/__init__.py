from .nmp import FixedNMP
from .recnn import RecursiveSimple, RecursiveGated
MODEL_DICT = dict(
    nmp=FixedNMP,
    recs=RecursiveSimple,
    recg=RecursiveGated
)
