import sys, os, copy
import numpy as np
import matplotlib.pyplot as plt

from pymortestbed.linalg        import ScipySpLu as linsolv
from pymortestbed.optimization  import NewtonRaphson
from pymortestbed.ode import dirk1_prim_lin, dirk2_prim_lin, dirk3_prim_lin
from pymortestbed.ode import dirk1_prim_nl , dirk2_prim_nl , dirk3_prim_nl
from pymortestbed.ode import dirk1_sens    , dirk2_sens    , dirk3_sens
from pymortestbed.ode import dirk1_dual    , dirk2_dual    , dirk3_dual
from pymortestbed.ode import dirk1_funcl   , dirk2_funcl   , dirk3_funcl
from pymortestbed.ode.dirk_util import dirk_int_state

from pymortestbed.app.burg_fv_1d import construct_ode, extract_param_funcl

from setup import BurgersGeneratorOrig

## Float in filename
def float_in_filename(f):
    s = '{0:16.12f}'.format(f)
    s = s.replace('.', 'p').replace('-', 'm').rstrip('0').rstrip('p').strip()
    if s[-1] == 'p': s += '0'
    return s

## Nonlinear solver
infnorm = lambda x: np.max(np.abs(x))
nlsolv  = NewtonRaphson(linsolv, 25, 1.0e-8, lambda x: infnorm(x))

## Temporal discretization
nstep = 500
T     = [0.0, 35.0]
t     = np.linspace(T[0], T[1], nstep+1)

## Spatial discretization
lmin, lmax, nel = 0.0, 100.0, 1000
gen = BurgersGeneratorOrig(lmin, lmax, nel)
U0 = np.ones(nel, dtype=float)

## Parametrization/Functional
p0 = np.array([[0.05, 0.02, 5.0]])
which_param, which_funcl = 'dbc', 'rightval'
nmu, dvelo_mu, nfuncl, funcl, dfuncl, dfuncl_mu = extract_param_funcl(
                                                       which_param, which_funcl)
## ODE 
ode_prim, ode_sens, ode_dual = construct_ode('dbc', 'rightval')
mass    , velo    , dvelo      = ode_prim.mass, ode_prim.velo, ode_prim.dvelo

## ODE Solver
nstage = 1
if nstage == 1:
    dirk_lin , dirk_nl   = dirk1_prim_lin, dirk1_prim_nl
    dirk_sens, dirk_dual = dirk1_sens    , dirk1_dual
    dirk_funcl           = dirk1_funcl
elif nstage == 2:
    dirk_lin , dirk_nl   = dirk2_prim_lin, dirk2_prim_nl
    dirk_sens, dirk_dual = dirk2_sens    , dirk2_dual
    dirk_funcl           = dirk2_funcl
elif nstage == 3:
    dirk_lin , dirk_nl   = dirk3_prim_lin, dirk3_prim_nl
    dirk_sens, dirk_dual = dirk3_sens    , dirk3_dual
    dirk_funcl           = dirk3_funcl

## Forward Time integration
for j in range(p0.shape[0]):
    U = np.zeros((nel, nstep+1), dtype=float, order='F')
    for i, ti in enumerate(t[:-1]):
        dt = t[i+1] - ti
        if np.mod(i, round(len(t)/10)) == 0:
            print('>>>>> Timestep {0:d} <<<<<'.format(i))
    
        # Primal
        gen.freeze_param(p0[j, :])
        #msh,phys,disc=gen.give_me()
        #print(msh.get_dual_nodes())        
        #print(phys.dbc)   
        #import matplotlib.pyplot as plt
        #plt.plot(msh.get_dual_nodes(),phys.src)
        #plt.show()
        #import sys
        #sys.exit()
        
        U[..., i+1], _ = dirk_nl(mass, velo, dvelo, U[..., i], None,
                                 ti, dt, gen, nlsolv)
    
    # Save snapshots
    mu_str = [float_in_filename(p0[j, 0]), float_in_filename(p0[j, 1]),
              float_in_filename(p0[j, 2])]
    np.savetxt('burg/snaps_{0:s}_{1:s}_{2:s}.dat'.format(*mu_str), U)
    
    ## Plot
    msh, _, _ = gen.give_me()
    
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    for i in np.round(np.linspace(0, nstep, 10)):
        plt.plot(msh.get_dual_nodes(), U[:, i], lw=2)
    plt.show()
