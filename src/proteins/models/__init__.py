from .wangnet import WangNet
#from .graphgen import GraphGen
from .GraphNetwork import ProteinGraphNetwork
#from .graphattention import GraphAttention

MODEL_DICT = dict(
    w=WangNet,
    #g=GraphGen,
    g=ProteinGraphNetwork
)
