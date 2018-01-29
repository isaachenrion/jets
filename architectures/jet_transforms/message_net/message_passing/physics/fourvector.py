import sys
import math
import torch

METRIC = [1,-1,-1,-1]

def contract_tuples(lhs,rhs,metric = None):
    return sum(m*l*r for m,l,r in zip(metric if metric else [1]*len(lhs),lhs,rhs))

def contract(lhs, rhs):
    return contract_tuples(lhs.components(),rhs.components(),METRIC)

class FourVector(object):
    def __init__(self,x0,x1,x2,x3,pytorch=False):
        self._x0 = x0
        self._x1 = x1
        self._x2 = x2
        self._x3 = x3
        self.pytorch = pytorch

    def __add__(lhs,rhs):
        return FourVector(*[sum(x) for x in zip(lhs.components(),rhs.components())])

    @property
    def x0(self):
        return self._x0

    @property
    def x1(self):
        return self._x1

    @property
    def x2(self):
        return self._x2

    @property
    def x3(self):
        return self._x3

    @property
    def eta(self):
        if abs(self.perp()) < sys.float_info.epsilon:
            return float('inf') if self.x3 >=0 else float('-inf')
        if self.pytorch:
            return -torch.log(torch.tan(self.theta/2.))
        else:
            return -math.log(math.tan(self.theta/2.))

    @property
    def theta(self):
        if self.pytorch:
            return torch.atan2(self.perp(),self.x3)
        else:
            return math.atan2(self.perp(),self.x3)

    @property
    def phi(self):
        if self.pytorch:
            return torch.atan2(self.x2,self.x1)
        else:
            return math.atan2(self.x2,self.x1)

    def components(self):
        return (self.x0,self.x1,self.x2,self.x3)

    def s2(self):
        return contract(self,self)

    def s(self):
        if self.pytorch:
            return torch.sqrt(contract(self,self))
        else:
            return math.sqrt(contract(self,self))

    def perp2(self):
        transvers_comps = self.components()[1:-1]
        return contract_tuples(transvers_comps,transvers_comps)

    def perp(self):
        if self.pytorch:
            return torch.sqrt(self.perp2())
        else:
            return math.sqrt(self.perp2())

class FourMomentum(FourVector):
    def __add__(lhs,rhs):
        return FourMomentum(*FourVector.__add__(lhs,rhs).components())

    e  = FourVector.x0

    px = FourVector.x1

    py = FourVector.x2

    pz = FourVector.x3

    @property
    def pt(self):
        return super(FourMomentum,self).perp()

    def m(self):
        return super(FourMomentum,self).s()

    def m2(self):
        return super(FourMomentum,self).s2()

    def pt2(self):
        return super(FourMomentum,self).perp2()

class FourPosition(FourVector):
    def __add__(lhs,rhs):
        return FourPosition(*FourVector.__add__(lhs,rhs).components())

    t = FourVector.x0

    x = FourVector.x1

    y = FourVector.x2

    z = FourVector.x3
