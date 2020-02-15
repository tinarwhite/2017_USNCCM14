import sys, os
import numpy as np

from pymortestbed.core import Mesh1d, Mesh1dUniform
from pymortestbed.core import Physics, Discretization, Generator

class BurgersGeneratorNonUni(Generator):
    def __init__(self, nodes):
        self.msh = Mesh1d(nodes)
        self.disc = Discretization()
        self.phys = Physics()
    def freeze_param(self, mu):
        self.mu = mu
        self.phys.src = mu[0]*np.exp(mu[1]*self.msh.get_dual_nodes())
        self.phys.dbc  = self.mu[2]
        return self
    def freeze_time(self, t):
        self.t         = t
        return self
    def set_mask(self,p):
        self.msh.p = p.copy('F')

class BurgersGenerator(Generator):
    def __init__(self, lmin, lmax, nel):
        self.msh = Mesh1dUniform(lmin, lmax, nel)
        self.disc = Discretization()
        self.phys = Physics()
    def freeze_param(self, mu):
        self.mu = mu
        self.phys.src = mu[0]*np.exp(mu[1]*self.msh.get_dual_nodes())
        self.phys.dbc  = self.mu[2]
        return self
    def freeze_time(self, t):
        self.t         = t
        return self
    def set_mask(self,p):
        self.msh.p = p.copy('F')

class BurgersGeneratorOrig(Generator):
    def __init__(self, lmin, lmax, nel):
        self.msh = Mesh1dUniform(lmin, lmax, nel)
        self.disc = Discretization()
        self.phys = Physics()
    def freeze_param(self, mu):
        self.mu = mu
        self.phys.src = mu[0]*np.exp(mu[1]*self.msh.get_dual_nodes())
        return self
    def freeze_time(self, t):
        self.t         = t
        self.phys.dbc  = self.mu[2]
        return self