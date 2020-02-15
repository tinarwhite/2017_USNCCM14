import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
import bisect
#from burg import plot_it
from cfd import plot_it
from util import read_me

def cluster_me(X, k):
    ck, ind, _ = cluster.k_means(X.T, k)
    Xk = [X[:, ind==i] for i in range(k)]
    return ck, Xk, ind

def pod(Xk, ny):
    Uk, Sk, Vk = np.linalg.svd(Xk, full_matrices=0, compute_uv=1)
    return Uk[:, :min(ny, Uk.shape[1])]
    
def simple_pod(Xk):
    Uk, Sk, Vk = np.linalg.svd(Xk, full_matrices=0, compute_uv=1)
    return Uk, Sk

def project(Vk, v):
    return np.dot(Vk, np.dot(Vk.T, v))

#matrix_interp(loc_new[k],loc_new[kk])
#(loc2,loc1) = (loc_new[0],loc_new[1])
def matrix_interp(loc2,loc1):
    A = np.zeros((len(loc2),len(loc1)))
    loc2 = list(loc2)
    loc1 = list(loc1)
    for i in range(len(loc2)):
        if loc2[i] in loc1:
            A[i,loc1.index(loc2[i])] = 1
        else:
            i1 = bisect.bisect(loc1, loc2[i])-1
            i2 = bisect.bisect(loc1, loc2[i])
            if i2 == len(loc1):
                i1 -= 1
                i2 -= 1
            if i2 == 0:
                i1 += 1
                i2 += 1
            A[i,i1] = 1-(loc2[i]-loc1[i1])/(loc1[i2]-loc1[i1])
            A[i,i2] = (loc2[i]-loc1[i1])/(loc1[i2]-loc1[i1])
    return A   
    
def downsamp_cberg(V,n):
    k = V[1].shape[0]
    eta = np.argmax(abs(V[:,0]))
    ind = [eta]
    nta = np.ceil(n/k).astype(int)
    for ivec in range(0,k):
        U = V[:,:ivec]
        for inode in range(nta):
            Vivec = V[:,ivec]
            Vbivec = Vivec[ind]
            #UbT = np.linalg.pinv(U[ind])
            UbT = U[ind].T
            Vtivec = np.dot(U,np.dot(UbT,Vbivec))
            Vtest = abs(Vivec-Vtivec)
            Vtest[ind] = 0
            eta = np.argmax(Vtest)
            ind.append(eta)
            ind = list(set(ind))
    return ind
    
def downsamp_old(A, loc, tol):
    nodes_ind = [0,1]
    ind = range(0,4)
    while ind[-1] < A.shape[0]:
         A_diff = np.zeros(A[ind[0]].shape)
         for i in ind[1:-1]:
              A_ind_est=A[ind[0]]+(A[ind[-1]]-A[ind[0]])*(loc[i]-loc[ind[0]])/(loc[ind[-1]]-loc[ind[0]])
              A_diff += np.square(A_ind_est - A[i])
         if np.max(A_diff) > tol:
             nodes_ind.append(ind[-1]-1)
             ind = range(ind[-1]-2,ind[-1]+2)
         else:
             ind = range(ind[0],ind[-1]+2)
    if nodes_ind[-1] != ind[-1]-1:
         nodes_ind.append(ind[-1]-1)
    nodes_ind.append(ind[-1])
    return nodes_ind

def downsamp_1D(A, loc, tol):
    nodes_ind = [0, 1]
    ind = range(0,4) # 0,1,2
    #count_loops = 0
    switch = 'next'
    while ind[-1] < A.shape[0]:
         A_diff = np.zeros(A[ind[0]].shape)
         for i in ind[1:-1]:
              A_ind_est=A[ind[0]]+(A[ind[-1]]-A[ind[0]])*(loc[i]-loc[ind[0]])/(loc[ind[-1]]-loc[ind[0]])
              A_diff += np.square(A_ind_est - A[i])
         maxdif = np.max(A_diff)
         if maxdif > tol and switch != 'forward' and ind[-1]-ind[0] > 3:
             ind = range(ind[0],ind[-1]) # same to one less
             switch = 'backward'
         elif maxdif < tol and switch != 'backward':
             ind = range(ind[0],ind[-1]+2) # same to one more
             switch = 'forward'
         else:
             nodes_ind.append(ind[-1]-1)
             ind = range(ind[-1]-2,ind[-1]+ind[-1]-ind[0]-1) # next to one more (remembering count)
             switch = 'next'
         #count_loops += 1
    if nodes_ind[-1] != ind[-1]-1:
         nodes_ind.append(ind[-1]-1)
    #print('count_loops = ',count_loops)
    #nodes_ind.append(ind[-1])
    return nodes_ind

    
def downsamp_smooth(A, nodes, tol):
    max_skew = 1.1
    dx = nodes[1]-nodes[0]
    loc = 0.5*(nodes[:-1]+nodes[1:])
    A_ind = list(range(0,A.shape[0]))
    A_test = np.ones_like(A_ind).astype(int)
    skip = 0
    zone = 'on'
    while zone == 'on':
        zone = 'off'
        zone_start = A_ind[0]
        zone_end = A_ind[-1]
        ind = range(zone_start,zone_start+2+skip)
        skip += 1
        while ind[-1] < zone_end+1:
             A_diff = 0
             for i in ind[1:-1]:
                  A_ind_est=A[ind[0]]+(A[ind[-1]]-A[ind[0]])*(loc[i]-loc[ind[0]])/(loc[ind[-1]]-loc[ind[0]])
                  A_diff += np.square(A_ind_est - A[i])
             #if np.sum(A_diff) < tol and np.max(A_diff) < tol/10:
             #if np.sum(A_diff) < tol:
             if np.max(A_diff) < tol/10:
                 A_test[ind[0]:ind[-1]+1] = skip
                 zone = 'on'
             ind = range(ind[-1],ind[-1]+skip+1)
                 
    switch = 'none'             
    switch_old = 'none'
    topologies = []
    A_ind_switches = []
    for i in range(A_test.shape[0]-1):
        if A_test[i+1] > A_test[i]:
            switch = 'increasing'
        if A_test[i+1] < A_test[i]:
            switch = 'decreasing'
        if switch != switch_old:
            topologies.append(switch)
            if switch_old == 'none':
                A_ind_switches.append(0)
            else:
                A_ind_switches.append(i)
        switch_old = switch
    A_ind_switches.append(i+1)
    
    guides = []
    ind = 0
    for ind_switch in A_ind_switches[:-1]:
        if ind_switch == 0 and topologies[0] == 'decreasing':
            guides.append('left_wall')
        if ind_switch != A_ind_switches[-2] and topologies[ind] == 'increasing':
            if A_test[A_ind_switches[ind]] < A_test[A_ind_switches[ind+2]]:
                guides.append('left_mound')
            else:
                guides.append('right_mound') 
        if ind_switch == A_ind_switches[-2] and topologies[-1] == 'increasing':
            guides.append('right_wall')
        ind +=1
        minimum = A_ind_switches[np.argmin(A_test[A_ind_switches])]

    A_fill = np.zeros_like(A_ind).astype(float)
    
    if guides == [] or np.sum(A_test[5:-5] - np.ones_like(A_test[5:-5])) < len(A_test[5:-5])/5:
        A_fill = loc
    for guide in guides:
        if guide == 'left_wall':
            del A_ind_switches[0]
            ind = A_ind_switches[0]
            skipmax = A_test[ind]*dx
            curloc = loc[ind]
            while ind > 0:
                skipmax = np.minimum(A_test[ind]*dx,max_skew*skipmax)
                A_fill[ind] = curloc
                curloc -= skipmax
                ind = (np.abs(loc-curloc)).argmin()
        if guide == 'left_mound':
            ind = A_ind_switches[0]
            skipmax = A_test[ind]*dx
            curloc = loc[ind]
            while ind < A_ind_switches[1]:
                skipmax = np.minimum(A_test[ind]*dx,max_skew*skipmax)
                A_fill[ind] = curloc
                curloc += skipmax
                ind = (np.abs(loc-curloc)).argmin()
            if skipmax < A_test[A_ind_switches[2]]*dx:
                while ind < A_ind_switches[2]:
                    skipmax = np.minimum(A_test[ind]*dx,max_skew*skipmax)
                    A_fill[ind] = curloc
                    curloc += skipmax
                    ind = (np.abs(loc-curloc)).argmin()
            else:
                ind = A_ind_switches[2]
                skipmax = np.minimum(A_test[A_ind_switches[2]]*dx,skipmax)
                curloc = loc[ind]
                while ind > A_ind_switches[1]:
                    skipmax = np.minimum(A_test[ind]*dx,max_skew*skipmax)
                    A_fill[ind] = curloc
                    curloc -= skipmax
                    ind = (np.abs(loc-curloc)).argmin()
            del A_ind_switches[0:2]
        if guide == 'right_mound':
            ind = A_ind_switches[2]
            skipmax = A_test[ind]*dx
            curloc = loc[ind]
            while ind > A_ind_switches[1]:
                skipmax = np.minimum(A_test[ind]*dx,max_skew*skipmax)
                A_fill[ind] = curloc
                curloc -= skipmax
                ind = (np.abs(loc-curloc)).argmin()
            if skipmax < A_test[A_ind_switches[0]]*dx:
                while ind > A_ind_switches[0]:
                    skipmax = np.minimum(A_test[ind]*dx,max_skew*skipmax)
                    A_fill[ind] = curloc
                    curloc -= skipmax
                    ind = (np.abs(loc-curloc)).argmin()
            else:
                ind = A_ind_switches[0]
                skipmax = np.minimum(A_test[A_ind_switches[2]]*dx,skipmax)
                curloc = loc[ind]
                while ind < A_ind_switches[1]:
                    skipmax = np.minimum(A_test[ind]*dx,max_skew*skipmax)
                    A_fill[ind] = curloc
                    curloc += skipmax
                    ind = (np.abs(loc-curloc)).argmin()
            del A_ind_switches[0:2]
        if guide == 'right_wall':
            ind = A_ind_switches[0]
            skipmax = A_test[ind]*dx
            curloc = loc[ind]
            while ind < A_ind_switches[-1]:
                skipmax = np.minimum(A_test[ind]*dx,max_skew*skipmax)
                A_fill[ind] = curloc
                curloc += skipmax
                ind = (np.abs(loc-curloc)).argmin()
    loc_new = np.array([value for value in A_fill if value != 0])
    nodes_new = np.zeros(loc_new.shape[0]+1)
    nodes_new[1:-1] = 0.5*(loc_new[1:]+loc_new[:-1])
    nodes_new[0] = nodes[0]
    nodes_new[-1] = nodes[-1]  
    return nodes_new
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    snaps =  read_me('burg/snaps_0p02_0p02_1.dat',
                     'burg/snaps_0p02_0p02_2p5.dat',
                     'burg/snaps_0p02_0p02_5.dat')
    #snaps =  read_me('cfd/naca0012ref0p1.snaps.mu2',
    #                 'cfd/naca0012ref0p1.snaps.mu6')

    nclust = 2
    ck, Xk = cluster_me(snaps, nclust)
    Vk = [pod(Xk[k], 10) for k in range(nclust)]

    # plot centers
    plot_it(ck[:, 0])

    # plot 1st 5 modes of 1st basis
    #plot_it(Vk[0][:, :5])
    plot_it(Vk[0][:, 0])

    # plot vector and its projection onto 1st basis
    #plot_it(np.hstack((snaps[:, [150]], project(Vk[0], snaps[:, [150]]))))
