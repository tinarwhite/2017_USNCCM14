import sys, os, copy
import numpy as np
import scipy
import matplotlib.pyplot as plt

#from pymortestbed.linalg        import ScipySpLu as linsolv
#from pymortestbed.linalg        import ScipyGmres as linsolv
#from pymortestbed.linalg        import ScipyLstSq as linsolv
from pymortestbed.linalg        import ScipySpLstSq as linsolv
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
        return ( sV )
    elif op in ['rmult']:
        #return ( np.dot(V, args[0]) )
        return ( scipy.sparse.csr_matrix.dot(sV, args[0]) )
    elif op == 'scale_add':
        return ( sV*args[1] + args[0] )
    else:
        raise ValueError('Operation "{0:s}" not implemented'.format(op))

def velo_new(y, Up, msh, phys, disc, op='return', *args):
    #return velo(np.dot(V, y), Up, msh, phys, disc, op, *args)
    return velo(scipy.sparse.csr_matrix.dot(sV, y), Up, msh, phys, disc, op, *args)

def dvelo_new(y, Up, msh, phys, disc, op='return', *args):
    #J = dvelo(np.dot(V, y), Up, msh, phys, disc, op, *args)
    J = dvelo(scipy.sparse.csr_matrix.dot(sV, y), Up, msh, phys, disc, op, *args)
    #return J.dot(V)
    return scipy.sparse.csr_matrix.dot(J,sV)
    
    

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
## Temporal discretization
nstep = 500 #500
T     = [0.0, 35.0]
t     = np.linspace(T[0], T[1], nstep+1)

# Proper POD with Col-Row clustering (Time-Space)
# Start with col clustering
from rom_stuff import pod, read_me, cluster_me, simple_pod
snaps =  read_me('burg/snaps_0p05_0p02_5.dat')
nclust = (5,10)
npod = 50
ck, Xk_col, ind_col = cluster_me(snaps, nclust[0])
#npod = [np.min(Xk_col[i].shape[1]) for i in range(nclust[0])]

# Now do row clustering and pod
V_crw = [1]*nclust[0]
sV_crw = [1]*nclust[0]
Sk = [1]*nclust[0]
Vkik = [1]*nclust[1]
Skik = [1]*nclust[1]
for i in range(nclust[0]):
    cki, Xki, indi = cluster_me(Xk_col[i].T, nclust[1])
    npod2 = np.hstack(([0],np.cumsum([np.min(Xki[k].shape) for k in range(nclust[1])])))
    Vki = np.zeros((snaps.shape[0],npod2[nclust[1]]),dtype=float) 
    for k in range(nclust[1]):
        Vkik[k], Skik[k] = simple_pod(Xki[k].T)
        Vki[indi==k,npod2[k]:npod2[k+1]] = Vkik[k]
    Ski = np.concatenate([Skik[k] for k in range(nclust[1])])
    indik = np.argsort(-Ski)
    Sk[i] = [Ski[i] for i in indik]
    V_crw[i] = np.array([Vki[:,i] for i in indik])[:npod,:].T
    sV_crw[i] = scipy.sparse.csr_matrix(V_crw[i]) 
        

## Spatial discretization
lmin, lmax, nel = 0.0, 100.0, snaps.shape[0] #0.0, 100.0, 1000
nodes = np.linspace(lmin, lmax, nel+1)
print([len(nodes) for k in range(nclust[0])])
gen = BurgersGeneratorNonUni(nodes)
#gen = BurgersGenerator(lmin, lmax, nel)
U0 = np.ones(nel, dtype=float)

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
    # for each time step, compute dist to nearest basis and use that basis
    # compute distance between U and Ck
    # So start with U0 and then use dist(V1*y1-ck)
    #ones might be better than zeros
    # and U0 (ones) projected onto V1 is better (if switch cluster, project
    # solution into your new subspace) project U0 into V[whatever] (dot prod)
    # V[1].T*U0 for example if [1] is the new cluster
    # But odn't use U0 later, use V[last k]*y[n-1] instead of U0
    # so you'll get V[new k].T*V[old k]*y[n-1]
    # where it's all dot products
    
    kclust = np.argmin([scipy.spatial.distance.euclidean(U0,ck[k]) for k in range(nclust[0])])
#    y = np.zeros((npod[0][0]*nclust[1], nstep+1), dtype=float, order='F')
    #V = V_crw[kclust] 
    sV = sV_crw[kclust] 
    y = np.zeros((sV_crw[kclust].shape[1], nstep+1), dtype=float, order='F')
    k_ind = np.zeros(y.shape[1], dtype=np.int)
    k_ind[0] = kclust 
    start_time = time.time()
    for i, ti in enumerate(t[:-1]):
        dt = t[i+1] - ti
        if np.mod(i, round(len(t)/10)) == 0:
            print('>>>>> Timestep {0:d} <<<<<'.format(i))
    
        # Primal
        gen.freeze_param(p0[j, :])
        y[..., i+1], _ = dirk_nl(mass_new, velo_new, dvelo_new, y[..., i], None,
                                 ti, dt, gen, nlsolv)
        #kclust = np.argmin([scipy.spatial.distance.euclidean(V.dot(y[:,i+1]),ck[k]) for k in range(nclust[0])])
        kclust = np.argmin([scipy.spatial.distance.euclidean(scipy.sparse.csr_matrix.dot(sV, y[:,i+1]),ck[k]) for k in range(nclust[0])])        
        k_ind[i+1] = kclust 
        if k_ind[i] != k_ind[i+1]:
            #print('swap')
            #y[..., i+1] = V_crw[kclust].T.dot(V).dot(y[..., i+1]) 
            y[..., i+1] = scipy.sparse.csr_matrix.dot(scipy.sparse.csr_matrix.dot(sV_crw[kclust].T, sV), y[..., i+1]) 
        #V = V_crw[kclust]
        sV = sV_crw[kclust]
    print("--- dsamp %s seconds ---" % round((time.time() - start_time),4))
    # Plot
    msh, _, _ = gen.give_me() 
    msh_plot = msh.get_dual_nodes()
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    for i in np.linspace(0, nstep, 15).astype(int):
        #proj = np.dot(V_crw[k_ind[i]], y[:,i])
        proj = scipy.sparse.csr_matrix.dot(sV_crw[k_ind[i]], y[:,i])
        plt.plot(msh_plot, proj, 'k-', lw=2)
        plt.plot(msh_plot, snaps[:,i], 'b--', lw=2)
    plt.show()
    
# Calculate RMS Error between snaps and ROM
    snaps_ROM = np.ones_like(snaps)
    for i in np.linspace(0, nstep, nstep+1).astype(int):
        #proj = np.dot(V_crw[k_ind[i]], y[:,i])
        proj = scipy.sparse.csr_matrix.dot(sV_crw[k_ind[i]], y[:,i])
        snaps_ROM[:,i] = proj
    #RMS_Error = np.sum(np.square(snaps_ROM-snaps))
    Mean_Sqrt_Error = np.sqrt(np.mean(np.square(snaps_ROM-snaps)))
    Mean_Abs_Error = np.mean(np.abs(snaps_ROM-snaps))
    
    
#print("--- RMS Error is %s---" %round(RMS_Error,5))
print("--- Mean Abs Error is %s---" %round(Mean_Abs_Error,5))
print("--- Mean Sqrt Error is %s---" %round(Mean_Sqrt_Error,5))

print('basis vectors:',[V_crw[i].shape[1] for i in range(nclust[0])])