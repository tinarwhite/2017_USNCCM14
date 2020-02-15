import time
#start_ind_time = time.time()

import sys, os, copy
import numpy as np
import scipy
import matplotlib.pyplot as plt

from pymortestbed.linalg        import ScipyLstSq as linsolv
from pymortestbed.optimization  import GaussNewton
from pymortestbed.ode import dirk1_prim_lin, dirk2_prim_lin, dirk3_prim_lin
from pymortestbed.ode import dirk1_prim_nl , dirk2_prim_nl , dirk3_prim_nl
from pymortestbed.ode import dirk1_sens    , dirk2_sens    , dirk3_sens
from pymortestbed.ode import dirk1_dual    , dirk2_dual    , dirk3_dual
from pymortestbed.ode import dirk1_funcl   , dirk2_funcl   , dirk3_funcl
from pymortestbed.ode.dirk_util import dirk_int_state 

from pymortestbed.app.burg_fv_1d import construct_ode, extract_param_funcl
from pymortestbed.app.burg_fv_1d.invisc import mass, velo, dvelo

from setup import BurgersGenerator, BurgersGeneratorNonUni

## Float in filename
def float_in_filename(f):
    s = '{0:16.12f}'.format(f)
    s = s.replace('.', 'p').replace('-', 'm').rstrip('0').rstrip('p').strip()
    if s[-1] == 'p': s += '0'
    return s

def mass_new(y, Up, msh, phys, disc, op='return', *args):
    if op in ['return', 'return_inv', 'return_transpose',
              'return_inv_transpose']:
        return ( V )
    elif op in ['rmult']:
        return ( np.dot(V, args[0]) )
    elif op == 'scale_add':
        return ( V*args[1] + args[0] )
    else:
        raise ValueError('Operation "{0:s}" not implemented'.format(op))

def velo_new(y, Up, msh, phys, disc, op='return', *args):
    return velo(np.dot(V, y), Up, msh, phys, disc, op, *args)

def dvelo_new(y, Up, msh, phys, disc, op='return', *args):
    J = dvelo(np.dot(V, y), Up, msh, phys, disc, op, *args)
    return J.dot(V)
    
### Hyper reduction changes

def mass_new_hrom(y, Up, msh, phys, disc, op='return', *args):
    if op in ['return', 'return_inv', 'return_transpose',
              'return_inv_transpose']:
        return ( V[msh.p,:] )
    elif op in ['rmult']:
        return ( np.dot(V[msh.p,:], args[0]) )
    elif op == 'scale_add':
        return ( V[msh.p,:]*args[1] + args[0] )
    else:
        raise ValueError('Operation "{0:s}" not implemented'.format(op))

def velo_new_hrom(y, Up, msh, phys, disc, op='return', *args):
    return velo(np.dot(V, y), Up, msh, phys, disc, op, *args)[msh.p]

def dvelo_new_hrom(y, Up, msh, phys, disc, op='return', *args):
    J = dvelo(np.dot(V, y), Up, msh, phys, disc, op, *args)
    return J[msh.p,:].dot(V)

## Nonlinear solver
infnorm = lambda x: np.max(np.abs(x))
lmult = lambda A, b: A.transpose().dot(b)
nlsolv  = GaussNewton(linsolv, 25, 1.0e-8, lambda x: infnorm(x),
                      lmult=lmult)
## Temporal discretization
nstep = 500 #500
T     = [0.0, 35.0]
t     = np.linspace(T[0], T[1], nstep+1)

# POD with Col-only clustering
from rom_stuff import pod, read_me, cluster_me, downsamp_old, downsamp_1D, matrix_interp
snaps =  read_me('burg/snaps_0p05_0p02_5.dat')#[::2]
nclust = 12 #7, 12
npod = [25]*nclust #40, 25
ck, Xk, ind = cluster_me(snaps, nclust)
Vk = [pod(Xk[k], npod[k]) for k in range(nclust)]
ny = [npod[k] for k in range(nclust)] #V.shape[1]

## Spatial discretization for entire mesh
lmin, lmax, nel = 0.0, 100.0, snaps.shape[0] #0.0, 100.0, 1000
nodes = np.linspace(lmin, lmax, nel+1)

# Hyper reduction, simple implementation
#A_ind =[list(np.arange(0,nodes.shape[0]-1,1))[::4] for k in range(nclust)]
#A_ind = [np.delete(np.arange(0,nodes.shape[0]-1,1),A_ind[k]) for k in range(nclust)]

# Hyper reduction at the chosen points
pct_tol = 1*10**-7 #1*10**-15 or -11 or -5
tol = pct_tol*(np.max(snaps)-np.min(snaps))
loc = 0.5*(nodes[:-1]+nodes[1:])
start_ind_time = time.time()
#A_ind = [downsamp_1D(Xk[k], loc, tol)[:-1] for k in range(nclust)]
A_ind = [downsamp_cberg(Vk[k], 164)[:-1] for k in range(nclust)]
print("--- cberg %s seconds ---" % round((time.time() - start_ind_time),4))
#print("--- cwhite %s seconds ---" % round((time.time() - start_ind_time),4))

# Hyper reduction additional parameters
print([len(A_ind[k])+1 for k in range(nclust)])
#Vk_down = [Vk[k][A_ind[k]] for k in range(nclust)]
Vk_down = Vk

#for k in range(nclust):
#    A_ind[k].append(snaps.shape[0])
#gen = [BurgersGeneratorNonUni(nodes[A_ind[k]]) for k in range(nclust)]
gen = [BurgersGeneratorNonUni(nodes) for k in range(nclust)]
mask_ind = []
for k, kgen in enumerate(gen):
    #kgen.set_mask(np.array(A_ind[k][:-1])+1)
    #mask_ind.append(np.delete(np.arange(0,nodes.shape[0]-1,1),A_ind[k]))
    mask_ind.append(np.array(A_ind[k]))
    kgen.set_mask(mask_ind[k])
    #print(kgen.msh.p)
U0 = [np.ones(len(A_ind[k]), dtype=float) for k in range(nclust)]
ck = [ck[k][np.array(A_ind[k])] for k in range(nclust)]


## Parametrization/Functional
p0 = np.array([[0.05, 0.02, 5.0]])
which_param, which_funcl = 'dbc', 'rightval'
nmu, dvelo_mu, nfuncl, funcl, dfuncl, dfuncl_mu = extract_param_funcl(
                                                       which_param, which_funcl)

## ODE 
#ode_prim, ode_sens, ode_dual = construct_ode('dbc', 'rightval')
#mass    , velo    , dvelo      = ode_prim.mass, ode_prim.velo, ode_prim.dvelo

## ODE Solver
nstage = 1
dirk_nl = dirk1_prim_nl

## Forward Time integration
for j in range(p0.shape[0]):
    kclust = np.argmin([scipy.spatial.distance.euclidean(U0[k],ck[k]) for k in range(nclust)])
    y = np.zeros((ny[kclust], nstep+1), dtype=float, order='F')
    V = Vk_down[kclust]
    k_ind = np.zeros(y.shape[1], dtype=np.int)
    k_ind[0] = kclust
    start_time = time.time()
    for i, ti in enumerate(t[:-1]): 
        dt = t[i+1] - ti
        if np.mod(i, round(len(t)/10)) == 0:
            print('>>>>> Timestep {0:d} <<<<<'.format(i))
    
        # Primal
        gen0 = gen[kclust]
        gen[kclust].freeze_param(p0[j, :])
        y[..., i+1], _ = dirk_nl(mass_new_hrom, velo_new_hrom, dvelo_new_hrom, y[..., i], None,
                                 ti, dt, gen[kclust], nlsolv)
        #kclust = np.argmin([scipy.spatial.distance.euclidean(V[A_ind[k]].dot(y[:,i+1]),ck[k]) for k in range(nclust)])
        kclust = np.argmin([np.mean(np.square(V[A_ind[k]].dot(y[:, i+1])-ck[k])) for k in range(nclust)])
        k_ind[i+1] = kclust
        if k_ind[i] != k_ind[i+1]:
            #print('swap')
            y[..., i+1] = Vk_down[kclust].T.dot(V).dot(y[..., i+1]) 
        V = Vk_down[kclust]
    print("--- dsamp %s seconds ---" % round((time.time() - start_time),4))
    # Plot
    msh_plot = []
    for k in range(nclust):
        msh, _, _ = gen[k].give_me()
        msh_plot.append(msh.get_dual_nodes())
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    for i in np.linspace(0, nstep, 15).astype(int):
        proj = np.dot(Vk_down[k_ind[i]], y)
        plt.plot(msh_plot[k_ind[i]], proj[:,i], 'k-', lw=2)
        plt.plot(nodes[:-1], snaps[:,i], 'b--', lw=2)
    plt.show()

# Calculate RMS Error between snaps and ROM
    snaps_HROM = np.ones_like(snaps)
    for i in np.linspace(0, nstep, nstep+1).astype(int):
        proj = np.dot(Vk_down[k_ind[i]], y[:,i])
        snaps_HROM[:,i] = proj
    #RMS_Error = np.sum(np.square(snaps_ROM-snaps))
    Mean_Sqrt_Error = np.sqrt(np.mean(np.square(snaps_HROM-snaps)))
    Mean_Abs_Error = np.mean(np.abs(snaps_HROM-snaps))

#print("--- RMS Error is %s---" %round(RMS_Error,5))
print("--- Mean HROM Abs Error is %s---" %round(Mean_Abs_Error,5))
print("--- Mean HROM Sqrt Error is %s---" %round(Mean_Sqrt_Error,5))
