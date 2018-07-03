from .wangnet import WangNet
from .graphgen import GraphGen
from .graphattention import GraphAttention

MODEL_DICT = dict(
    w=WangNet,
    g=GraphGen,
    a=GraphAttention
)
