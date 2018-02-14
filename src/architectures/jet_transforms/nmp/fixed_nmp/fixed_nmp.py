import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from .adjacency import construct_physics_based_adjacency_matrix

from ..stacked_nmp.attention_pooling import construct_pooling_layer
from ..message_passing import construct_mp_layer
from ..message_passing.adjacency import construct_adjacency_matrix_layer

from .....architectures.readout import construct_readout
from .....architectures.embedding import construct_embedding
from .....visualizing import visualize_batch_2D
from .....monitors import Histogram

class FixedAdjacencyNMP(nn.Module):
    def __init__(self,
        features=None,
        hidden=None,
        iters=None,
        readout=None,
        **kwargs
        ):
        super().__init__()
        self.iters = iters
        self.embedding = construct_embedding('simple', features + 1, hidden, act=kwargs.get('act', None))
        self.mp_layers = nn.ModuleList([construct_mp_layer('fixed', hidden=hidden,**kwargs) for _ in range(iters)])
        self.readout = construct_readout(readout, hidden, hidden)
        self.adjacency_matrix = self.set_adjacency_matrix(features=features, **kwargs)
        self.logger = kwargs.get('logger', None)
        self.dij_histogram = Histogram('dij', n_bins=10, rootname='dij', append=True)
        self.dij_histogram.initialize(None, self.logger.plotsdir)

    def set_adjacency_matrix(self, **kwargs):
        pass

    def forward(self, jets, mask=None, **kwargs):
        h = self.embedding(jets)
        dij = self.adjacency_matrix(jets, mask=mask)
        for mp in self.mp_layers:
            h, _ = mp(h=h, mask=mask, dij=dij, **kwargs)
        out = self.readout(h)

        # logging
        ep = kwargs.get('epoch', None)
        iters_left = kwargs.get('iters_left', None)
        if self.logger is not None:
            if ep is not None and ep % 1 == 0:
                self.dij_histogram(values=dij.view(-1))
                #print(ep, iters_left)
                if iters_left == 0:
                    self.dij_histogram.visualize('dij-{}'.format(ep))
                    self.dij_histogram.clear()
                    visualize_batch_2D(dij, self.logger, 'epoch{}/adjacency'.format(ep))

        return out, _

class PhysicsNMP(FixedAdjacencyNMP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_adjacency_matrix(self, **kwargs):
        return construct_physics_based_adjacency_matrix(
                        alpha=kwargs.pop('alpha', None),
                        R=kwargs.pop('R', None),
                        trainable_physics=kwargs.pop('trainable_physics', None)
                        )

class PhysicsPlusLearnedNMP(FixedAdjacencyNMP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def physics_component(self):
        return torch.sigmoid(self._physics_component)

    @physics_component.setter
    def physics_component(self, value):
        self._physics_component = torch.FloatTensor([float(value)])
        if self.learned_tradeoff:
            self._physics_component = nn.Parameter(self._physics_component)
        else:
            self._physics_component = Variable(self._physics_component)
            if torch.cuda.is_available():
                self._physics_component = self._physics_component.cuda()

    def set_adjacency_matrix(self, **kwargs):
        self.learned_tradeoff = kwargs.pop('learned_tradeoff', False)
        self.physics_component = kwargs.pop('physics_component', None)

        self.learned_matrix = construct_adjacency_matrix_layer(
                    kwargs.get('adaptive_matrix', None),
                    hidden=kwargs.get('features', None) + 1,
                    symmetric=kwargs.get('symmetric', None)
                    )

        self.physics_matrix = construct_physics_based_adjacency_matrix(
                        alpha=kwargs.pop('alpha', None),
                        R=kwargs.pop('R', None),
                        trainable_physics=kwargs.pop('trainable_physics', None)
                        )

        def combined_matrix(jets, **kwargs):
            out = self.physics_component * self.physics_matrix(jets, **kwargs) \
                + (1 - self.physics_component) * self.learned_matrix(jets, **kwargs)
            return out
        return combined_matrix

    #def forward(self, **kwargs):


class EyeNMP(FixedAdjacencyNMP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_adjacency_matrix(self, **kwargs):
        def eye(jets, **kwargs):
            bs, sz, _ = jets.size()
            matrix = Variable(torch.eye(sz).unsqueeze(0).repeat(bs, 1, 1))
            if torch.cuda.is_available():
                matrix = matrix.cuda()
            if mask is None:
                return matrix
            return mask * matrix
        return eye

class OnesNMP(FixedAdjacencyNMP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_adjacency_matrix(self, **kwargs):
        def ones(jets, mask=mask, **kwargs):
            bs, sz, _ = jets.size()
            matrix = Variable(torch.ones(bs, sz, sz))
            if torch.cuda.is_available():
                matrix = matrix.cuda()
            if mask is None:
                return matrix
            return mask * matrix
        return ones

class LearnedFixedNMP(FixedAdjacencyNMP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_adjacency_matrix(self, **kwargs):
        matrix = construct_adjacency_matrix_layer(
                    kwargs.get('adaptive_matrix', None),
                    hidden=kwargs.get('features', None) + 1,
                    symmetric=kwargs.get('symmetric', None)
                    )
        return matrix

class PhysicsStackNMP(nn.Module):
    def __init__(self,
        features=None,
        hidden=None,
        iters=None,
        readout=None,
        scales=None,
        mp_layer=None,
        pooling_layer=None,
        **kwargs
        ):
        super().__init__()
        self.iters = iters
        self.embedding = construct_embedding('simple', features + 1, hidden, act=kwargs.get('act', None))
        self.physics_nmp = PhysicsNMP(features, hidden, 1, readout='constant', **kwargs)
        self.readout = construct_readout(readout, hidden, hidden)
        self.attn_pools = nn.ModuleList([construct_pooling_layer(pooling_layer, scales[i], hidden) for i in range(len(scales))])
        self.nmps = nn.ModuleList([construct_mp_layer(mp_layer, hidden=hidden, **kwargs) for _ in range(len(scales))])

    def forward(self, jets, mask=None, **kwargs):
        h, _ = self.physics_nmp(jets, mask, **kwargs)
        for pool, nmp in zip(self.attn_pools, self.nmps):
            h = pool(h)
            h, A = nmp(h=h, mask=None, **kwargs)
        out = self.readout(h)
        return out, A
