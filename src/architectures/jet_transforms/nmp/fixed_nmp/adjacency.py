import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

#import visdom
#from monitors import Regurgitate
#viz = visdom.Visdom()
#alpha_monitor = Regurgitate('alpha', visualizing=True)
#alpha_monitor.initialize(None, None, viz)
#R_monitor = Regurgitate('R', visualizing=True)
#R_monitor.initialize(None, None, viz)

def construct_physics_based_adjacency_matrix(alpha=None, R=None, trainable_physics=False):
    if trainable_physics:
        #assert R is None
        return TrainablePhysicsBasedAdjacencyMatrix(alpha_init=alpha, R_init=R)
    else:
        return FixedPhysicsBasedAdjacencyMatrix(alpha=alpha, R=R)

def compute_dij(p, alpha, R):
    p1 = p.unsqueeze(1)
    p2 = p.unsqueeze(2)

    delta_eta = p1[:,:,:,1] - p2[:,:,:,1]

    delta_phi = p1[:,:,:,2] - p2[:,:,:,2]
    delta_phi = torch.remainder(delta_phi + math.pi, 2*math.pi) - math.pi

    delta_r = (delta_phi**2 + delta_eta**2)**0.5

    dij = torch.min(p1[:,:,:,0]**(2.*alpha), p2[:,:,:,0]**(2.*alpha)) * delta_r / R

    return dij

class _PhysicsBasedAdjacencyMatrix(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def alpha(self):
        pass

    @property
    def R(self):
        pass

    def forward(self, p, mask=None, **kwargs):
        dij = compute_dij(p, self.alpha, self.R)
        #out = torch.exp(-dij)
        #if mask is None:
        #    return out
        #return mask * out
        return F.softmax(-dij.transpose(0, -1)).transpose(0, -1)

class FixedPhysicsBasedAdjacencyMatrix(_PhysicsBasedAdjacencyMatrix):
    def __init__(self, alpha=None, R=None):
        super().__init__()
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

class RTrainablePhysicsBasedAdjacencyMatrix(_PhysicsBasedAdjacencyMatrix):
    def __init__(self, alpha):
        super().__init__()
        #self.alpha_raw = nn.Parameter(torch.Tensor([0]))
        self._alpha = Variable(torch.FloatTensor([alpha]))
        if torch.cuda.is_available():
            self._alpha = self._alpha.cuda()
        self.logR = nn.Parameter(torch.Tensor([0]))

    @property
    def alpha(self):
        #return 1
        return self._alpha
        #alpha = F.sigmoid(self.alpha_raw)
        #alpha_monitor(alpha=alpha.data.numpy())
        #return F.tanh(self.alpha_raw)

    @property
    def R(self):
        R = torch.exp(self.logR)
        #R_monitor(R=R.data[0])
        return R

class TrainablePhysicsBasedAdjacencyMatrix(_PhysicsBasedAdjacencyMatrix):
    def __init__(self, alpha_init=0, R_init=0):
        super().__init__()
        self.alpha_raw = nn.Parameter(torch.Tensor([alpha_init]))
        self.logR = nn.Parameter(torch.Tensor([R_init]))
        #self._R = Variable(torch.FloatTensor([R]))
        #if torch.cuda.is_available():
        #    self._R = self._R.cuda()

    @property
    def alpha(self):
        alpha = F.tanh(self.alpha_raw)
        #import ipdb; ipdb.set_trace()
        #alpha_monitor(alpha=alpha.data[0])
        return alpha

    @property
    def R(self):
        R = torch.exp(self.logR)
        #R_monitor(R=R.data[0])
        return R

'''
def calculate_dij(p1, p2, alpha=1., R=1.):
    delta_eta = p1[1] - p2[1]

    delta_phi = p1[2] - p2[2]
    while delta_phi < - math.pi:
        delta_phi += 2. * math.pi
    while delta_phi > math.pi:
        delta_phi -= 2. * math.pi

    delta_r = (delta_phi**2 + delta_eta**2)**0.5

    dij = min(p1[0]**(2.*alpha), p2[0]**(2.*alpha)) * delta_r / R

    return dij

def torch_calculate_dij(p, alpha=1., R=.1):
    p1 = p.unsqueeze(1)
    p2 = p.unsqueeze(2)

    delta_eta = p1[:,:,:,1] - p2[:,:,:,1]

    delta_phi = p1[:,:,:,2] - p2[:,:,:,2]
    delta_phi = torch.remainder(delta_phi + math.pi, 2*math.pi) - math.pi

    delta_r = (delta_phi**2 + delta_eta**2)**0.5

    dij = torch.min(p1[:,:,:,0]**(2.*alpha), p2[:,:,:,0]**(2.*alpha)) * delta_r / R

    return dij

def NEWcalculate_dij_matrices(jets_padded, mask, alpha=1., R=.1):
    dij_list = []

    for counter, j in enumerate(X):

        #if counter % 1000 == 0:
        #    print(counter, 'jets done')

        constituents_transformed = []
        for c in jets_padded:
            lv = FourMomentum(c, pytorch=True)
            constituents_transformed.append( (lv.pt(),lv.eta(),lv.phi()) )

        n_constituents = len(constituents_transformed)
        dij = np.zeros((n_constituents,n_constituents))

        for i, pi in enumerate(constituents_transformed):
            for j, pj in enumerate(constituents_transformed):
                dij[i,j] = calculate_dij(pi, pj, alpha=alpha, R=R)

        dij_list.append(dij)

    return dij_list

def calculate_dij_matrices(X, alpha=1., R=1.):
    dij_list = []

    for counter, j in enumerate(X):

        #if counter % 1000 == 0:
        #    print(counter, 'jets done')

        constituents = j["content"][j["tree"][:, 0] == -1]
        constituents_transformed = []
        for c in constituents:
            lv = FourMomentum(c)
            constituents_transformed.append( (lv.pt(),lv.eta(),lv.phi()) )

        n_constituents = len(constituents_transformed)
        dij = np.zeros((n_constituents,n_constituents))

        for i, pi in enumerate(constituents_transformed):
            for j, pj in enumerate(constituents_transformed):
                dij[i,j] = calculate_dij(pi, pj, alpha=alpha, R=R)

        dij_list.append(dij)

    return dij_list
'''
