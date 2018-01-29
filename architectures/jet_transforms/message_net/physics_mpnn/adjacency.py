#from .fourvector import FourMomentum
import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class PhysicsBasedAdjacencyMatrix(nn.Module):
    def __init__(self, alpha_raw=0, logR=0, trainable=False):
        super().__init__()
        if trainable:
            self.alpha_raw = nn.Parameter(torch.FloatTensor([alpha_raw]))
            self.logR = nn.Parameter(torch.FloatTensor([logR]))
        else:
            self.alpha_raw = Variable(torch.FloatTensor([alpha_raw]))
            self.logR = Variable(torch.FloatTensor([logR]))

    def alpha(self):
        return F.tanh(self.alpha_raw)

    def R(self):
        return torch.exp(self.logR)

    def forward(self, p):
        p1 = p.unsqueeze(1)
        p2 = p.unsqueeze(2)

        delta_eta = p1[:,:,:,1] - p2[:,:,:,1]

        delta_phi = p1[:,:,:,2] - p2[:,:,:,2]
        delta_phi = torch.remainder(delta_phi + math.pi, 2*math.pi) - math.pi

        delta_r = (delta_phi**2 + delta_eta**2)**0.5

        dij = torch.min(p1[:,:,:,0]**(2.*self.alpha()), p2[:,:,:,0]**(2.*self.alpha())) * delta_r / self.R()

        return dij

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
