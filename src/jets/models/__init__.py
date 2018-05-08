from .nmp import FixedNMP, VariableNMP
from .recnn import RecursiveSimple, RecursiveGated
MODEL_DICT = dict(
    nmp=FixedNMP,
    vnmp=VariableNMP,
    recs=RecursiveSimple,
    recg=RecursiveGated
)
