import sys, os, copy
import numpy as np
import scipy
import matplotlib.pyplot as plt

from pymortestbed.linalg        import ScipyLstSq as linsolv
from pymortestbed.linalg        import ScipySpLu as linsolv2
from pymortestbed.optimization  import GaussNewton, NewtonRaphson
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
    
### 

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

# SOLVE HDM -> snapshots (solv_uns.py)
# POD on snapshots -> V  (rom_stuff.py)
# SOLVE ROM              (solv_rom.py)

# Tasks
# 1) Proper POD (not stupid identity)
# 2) Clean up workflow
# 3) Row-only clustering
# 4) Column-only clustering
# 5) Row-column clustering
## Nonlinear solver
infnorm = lambda x: np.max(np.abs(x))
lmult = lambda A, b: A.transpose().dot(b)
nlsolv  = GaussNewton(linsolv, 25, 1.0e-8, lambda x: infnorm(x),
                      lmult=lmult)
nlsolv2  = NewtonRaphson(linsolv2, 25, 1.0e-8, lambda x: infnorm(x))

## Temporal discretization
nstep = 500 #500
T     = [0.0, 35.0]
t     = np.linspace(T[0], T[1], nstep+1)

# Proper POD with Row-only clustering
from rom_stuff import pod, read_me, cluster_me
snaps =  read_me('burg/snaps_0p05_0p02_5.dat')#[::2]
nclust = 4
npod = [50]*nclust
ck, Xk, ind = cluster_me(snaps, nclust)
Vk = [pod(Xk[k], npod[k]) for k in range(nclust)]
#V = np.eye(nel)

#V = np.eye(nel)[:, :200]
ny = npod[0] #V.shape[1]
## Spatial discretization
lmin, lmax, nel = 0.0, 100.0, snaps.shape[0] #0.0, 100.0, 1000
nodes = np.linspace(lmin, lmax, nel+1)

nodes2 = np.linspace(lmin, lmax, 10000)

A_ind =list(np.delete(np.arange(0,nodes2.shape[0],1), np.arange(1,nodes2.shape[0],2),0)) 
#Xk_down = [Xk[k][np.array(A_ind[k])] for k in range(nclust)]

#A_ind = [downsamp(Xk[k], nodes, tol) for k in range(nclust)]
nodes_new = nodes2[A_ind]


gen = BurgersGeneratorNonUni(nodes_new)
gen.set_mask(np.arange(0,nodes_new.shape[0]-1,dtype=int)[::2])
print(gen.msh.p)
#gen = BurgersGenerator(lmin, lmax, nel)
U0 = 0.0*np.ones(nodes_new.shape[0]-1, dtype=float)
U = 0.0*np.ones((nodes_new.shape[0]-1,501), dtype=float)


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

# Record time
import time

## Forward Time integration
for j in range(p0.shape[0]):
#    y = np.zeros((ny, nstep+1), dtype=float, order='F')
#    U0 = np.zeros((nodes.shape[0]-1, nstep+1), dtype=float, order='F')
    U[...,0]=U0
#    kclust = np.argmin([scipy.spatial.distance.euclidean(U0,ck[k]) for k in range(nclust)])
#    V = Vk[kclust] 
#    k_ind = np.zeros(y.shape[1], dtype=np.int)
#    k_ind[0] = kclust 
    start_time = time.time()
    for i, ti in enumerate(t[:-1]):
        dt = t[i+1] - ti
        if np.mod(i, round(len(t)/10)) == 0:
            print('>>>>> Timestep {0:d} <<<<<'.format(i))
        # Primal
#        def mass_new_1(y, Up, msh, phys, disc, op='return', *args):
#            return mass_new(V, y, Up, msh, phys, disc, op, *args)
#        def velo_new_1(y, Up, msh, phys, disc, op='return', *args):
#            return velo_new(V, y, Up, msh, phys, disc, op, *args)
#        def dvelo_new_1(y, Up, msh, phys, disc, op='return', *args):
#            return dvelo_new(V, y, Up, msh, phys, disc, op, *args)
        gen.freeze_param(p0[j, :])
 #       msh,phys,disc=gen.give_me()
        #print(phys.src)
        #import sys
        #sys.exit()
#        y[..., i+1], _ = dirk_nl(mass_new, velo_new, dvelo_new, y[..., i], None,
#                                 ti, dt, gen, nlsolv)
#        y[..., i+1], _ = dirk_nl(mass_new_1, velo_new_1, dvelo_new_1, y[..., i], None,
#                                 ti, dt, gen, nlsolv)
        #U[..., i+1], _ = dirk_nl(mass, velo, dvelo, U[..., i], None,
        #                         ti, dt, gen, nlsolv2)
        y[..., i+1], _ = dirk_nl(mass_new_hrom, velo_new_hrom, dvelo_new_hrom, y[..., i], None,
                                 ti, dt, gen, nlsolv)        
#        U[...,i+1] = np.dot(V,y[...,i+1])                         
#        kclust = np.argmin([scipy.spatial.distance.euclidean(V.dot(y[:,i+1]),ck[k]) for k in range(nclust)])
#        y[..., i+1] = Vk[kclust].T.dot(V).dot(y[..., i+1])  
#        k_ind[i+1] = kclust 
#        V = Vk[kclust]
    print("--- dsamp %s seconds ---" % round((time.time() - start_time),4))
    # Plot
    msh, _, _ = gen.give_me() 
    msh_plot = msh.get_dual_nodes()
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    for i in np.linspace(0, nstep, 15).astype(int):
        proj = np.dot(Vk[k_ind[i]], y)
        #proj = Vk[k_ind[i]].dot(Vk[k_ind[i]].T.dot(snaps))
        #plt.plot(msh_plot, proj[:,i], 'k-', lw=2)
        #plt.plot(msh_plot, U[:,i], 'k-', lw=2)        
        plt.plot(nodes[:-1], snaps[:,i], 'b--', lw=2)
    plt.show()
np.savetxt('test_save.dat',U)
