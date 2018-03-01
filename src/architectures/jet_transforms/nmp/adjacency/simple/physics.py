import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ._adjacency import _Adjacency


def construct_physics_adjacency(alpha=None, R=None, trainable_physics=False):
    if trainable_physics:
        assert R is None
        assert alpha is None
        return TrainablePhysicsAdjacency()
    else:
        return FixedPhysicsAdjacency(alpha=alpha, R=R)


def compute_dij(p, alpha, R):
    p1 = p.unsqueeze(1) + 1e-10
    p2 = p.unsqueeze(2) + 1e-10

    delta_eta = p1[:,:,:,1] - p2[:,:,:,1]

    delta_phi = p1[:,:,:,2] - p2[:,:,:,2]
    delta_phi = torch.remainder(delta_phi + math.pi, 2*math.pi) - math.pi

    delta_r = (delta_phi**2 + delta_eta**2)**0.5

    dij = torch.min(p1[:,:,:,0]**(2.*alpha), p2[:,:,:,0]**(2.*alpha)) * delta_r / R

    return dij

class _PhysicsAdjacency(_Adjacency):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    @property
    def alpha(self):
        pass

    @property
    def R(self):
        pass

    def raw_matrix(self, p, mask=None, **kwargs):
        dij = compute_dij(p, self.alpha, self.R)
        #dij = torch.exp(-dij)
        #import ipdb; ipdb.set_trace()
        return -dij


class FixedPhysicsAdjacency(_PhysicsAdjacency):
    def __init__(self, alpha=None, R=None,**kwargs):
        super().__init__(name='phy', **kwargs)
        self._alpha = Variable(torch.FloatTensor([alpha]))
        self._R = Variable(torch.FloatTensor([R]))
        if torch.cuda.is_available():
            self._alpha = self._alpha.cuda()
            self._R = self._R.cuda()

    @property
    def alpha(self):
        return self._alpha

    @property
    def R(self):
        return self._R


class TrainablePhysicsAdjacency(_PhysicsAdjacency):
    def __init__(self, alpha_init=0, R_init=0,**kwargs):
        super().__init__(name='tphy', **kwargs)
        base_alpha_init = 0
        base_R_init = 0

        #def artanh(x):
        #    assert torch.abs(x) < 1
        #    return 0.5 * torch.log((1 + x) / 1 - x)
        #alpha_init = artanh(alpha_init)

        self._base_alpha = nn.Parameter(torch.Tensor([base_alpha_init]))
        self._base_R = nn.Parameter(torch.Tensor([base_R_init]))
        #self._R = Variable(torch.FloatTensor([R]))
        #if torch.cuda.is_available():
        #    self._R = self._R.cuda()

    @property
    def alpha(self):
        alpha = F.tanh(self._base_alpha)
        #import ipdb; ipdb.set_trace()
        #alpha_monitor(alpha=alpha.data[0])
        return alpha

    @property
    def R(self):
        R = torch.exp(self._base_R)
        #R_monitor(R=R.data[0])
        return R

PHYSICS_ADJACENCIES = dict(
    tphy=TrainablePhysicsAdjacency,
    phy=FixedPhysicsAdjacency
)
