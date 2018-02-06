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
        assert R is None
        return TrainablePhysicsBasedAdjacencyMatrix(alpha)
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

    def alpha(self):
        pass

    def R(self):
        pass

    def forward(self, p):
        dij = compute_dij(p, self.alpha(), self.R())
        return torch.exp(-dij)

class FixedPhysicsBasedAdjacencyMatrix(_PhysicsBasedAdjacencyMatrix):
    def __init__(self, alpha=1, R=1):
        super().__init__()
        self._alpha = Variable(torch.FloatTensor([alpha]))
        self._R = Variable(torch.FloatTensor([R]))
        if torch.cuda.is_available():
            self._alpha = self._alpha.cuda()
            self._R = self._R.cuda()

    def alpha(self):
        return self._alpha

    def R(self):
        return self._R

class TrainablePhysicsBasedAdjacencyMatrix(_PhysicsBasedAdjacencyMatrix):
    def __init__(self, alpha):
        super().__init__()
        #self.alpha_raw = nn.Parameter(torch.Tensor([0]))
        self._alpha = Variable(torch.FloatTensor([alpha]))
        if torch.cuda.is_available():
            self._alpha = self._alpha.cuda()
        self.logR = nn.Parameter(torch.Tensor([0]))

    def alpha(self):
        #return 1
        return self._alpha
        #alpha = F.sigmoid(self.alpha_raw)
        #alpha_monitor(alpha=alpha.data.numpy())
        #return F.tanh(self.alpha_raw)

    def R(self):
        R = torch.exp(self.logR)
        #R_monitor(R=R.data[0])
        return R

class AlphaTrainablePhysicsBasedAdjacencyMatrix(_PhysicsBasedAdjacencyMatrix):
    def __init__(self, R):
        super().__init__()
        self.alpha_raw = nn.Parameter(torch.Tensor([1]))
        self._R = Variable(torch.FloatTensor([R]))
        if torch.cuda.is_available():
            self._R = self._R.cuda()

    def alpha(self):
        alpha = F.tanh(self.alpha_raw)
        import ipdb; ipdb.set_trace()
        alpha_monitor(alpha=alpha.data[0])
        return alpha

    def R(self):
        return self._R

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
