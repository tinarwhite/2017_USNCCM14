import numpy as np
from sklearn import cluster
from scipy import interpolate
import matplotlib.pyplot as plt
from burg import plot_it, locations
#from cfd import plot_it
#from cfd import plot_ind
from util import read_me

def cluster_me(X, k):
    ck, ind, _ = cluster.k_means(X.T, k)
    Xk = [X[:, ind==i] for i in range(k)]
    return ck, Xk, ind

def pod(Xk, ny):
    Uk, Sk, Vk = np.linalg.svd(Xk, full_matrices=0, compute_uv=1)
    return Uk[:, :min(ny, Uk.shape[1])]

def project(Vk, v):
    return np.dot(Vk, np.dot(Vk.T, v))

from rom_stuff import pod, read_me, cluster_me, downsamp_1D, downsamp_cberg, matrix_interp

#def reconstruct(A_downsamp, A_indices)
    
if __name__ == '__main__':
    snaps =  read_me('burg/snaps_0p05_0p02_5.dat')
#    snaps = read_me('cfd/naca0012ref0p1.snaps.mu0',
#                    'cfd/naca0012ref0p1.snaps.mu3',
#                    'cfd/naca0012ref0p1.snaps.mu5',
#                    'cfd/naca0012ref0p1.snaps.mu7',)

    #test with small number of snaps
#    snaps = snaps[0:6,0:9]

    # Apply k means clustering to samples
    nclusti = 6
    nclustj = 2
    nclust = (nclusti,nclustj)
    Xk = []
    ck = []
    ind = [0]*nclust[0]
    ck_col, Xk_col, ind_col = cluster_me(snaps, nclust[0])
    
    ## Spatial discretization for entire mesh
    lmin, lmax, nel = 0.0, 100.0, snaps.shape[0] #0.0, 100.0, 1000
    nodes = np.linspace(lmin, lmax, nel+1)
    
    # Hyper reduction at the chosen points
    pvect = 250
    pct_tol = 1*10**-7 #1*10**-15 or -11 or -5
    tol = pct_tol*(np.max(snaps)-np.min(snaps))
    loc = 0.5*(nodes[:-1]+nodes[1:])
    A_ind = [downsamp_1D(Xk_col[k], loc, tol)[:-1] for k in range(nclust[0])]
    lenA = [len(A_ind[k]) for k in range(nclust[0])]
    print(lenA)
    Xk_col_down = [Xk_col[k][np.array(A_ind[ind_col[pvect]])] for k in range(nclust[0])]
    
    # Perform POD on column clusters
    Vk_col = [pod(Xk_col[k], 10) for k in range(nclust[0])]    
    
    # Perform POD on downsampled column clusters
    Vk_col_down = [pod(Xk_col_down[k], 10) for k in range(nclust[0])]    
    
    # Hyper reduction according to carlberg at the chosen points    
    A_ind_cberg = [downsamp_cberg(Vk_col[k], 350) for k in range(nclust[0])]
    Xk_col_down_cberg = [Xk_col[k][np.array(np.sort(A_ind_cberg[ind_col[pvect]]))] for k in range(nclust[0])]
    Vk_col_down_cberg = [Vk_col[k][np.sort(A_ind_cberg[ind_col[pvect]])] for k in range(nclust[0])]
    Vk_col_down_cberg = [pod(Xk_col_down_cberg[k], 10) for k in range(nclust[0])] 
    
#    
#    # Apply k means clustering to deconstruct domain into submatrices
#    for i in range(nclust[0]):
#        cki, Xki, indi = cluster_me(Xk_col[i].T, nclust[1])
#        Xk = Xk + Xki
#        ck = list(ck)+list(cki)
#        ind[i] = indi
#    Vk = [pod(Xk[k].T, 50) for k in range(nclust[0]*nclust[1])]
#    Vk2 = [pod(Vk[k].T, 40) for k in range(nclust[0]*nclust[1])]
#    
#    #Choose a vector for testing
    #pvect=1050 #1050 for berg, 20 for CFD
    #pvect2 =200
    
    # Construct test vector in downsampled space
    downsample = snaps[:, pvect][np.array(A_ind[ind_col[pvect]])]
    downlocs = np.array(A_ind[ind_col[pvect]])/10
#    downsample2 = snaps[:, [pvect2]][np.array(A_ind[ind_col[pvect2]])]
#    downlocs2 = np.array(A_ind[ind_col[pvect2]])/10
    # Project Vk matrix onto given test vector in downsampled space
    proj = project(Vk_col_down[ind_col[pvect]], downsample)
    # Reconstruct vector in original space?
    projinterp = np.interp(np.linspace(0.0, 100.0, snaps.shape[0]),downlocs,proj)
    
    # Construct test vector in downsampled CBERG space
    Aind = np.sort(A_ind_cberg[ind_col[pvect]])
    downsample_cberg = snaps[:, pvect][np.array(Aind)]
    downlocs_cberg = np.array(Aind)/10
#    downsample2 = snaps[:, [pvect2]][np.array(A_ind[ind_col[pvect2]])]
#    downlocs2 = np.array(A_ind[ind_col[pvect2]])/10
    # Project Vk matrix onto given test vector in downsampled space
    proj_cberg = project(Vk_col_down_cberg[ind_col[pvect]], downsample_cberg)
    # Reconstruct vector in original space?
    #projinterp_cberg = np.interp(np.linspace(0.0, 100.0, snaps.shape[0]),downlocs_cberg,proj_cberg)
    f = interpolate.interp1d(downlocs_cberg, proj_cberg, fill_value = "extrapolate")
    projinterp_cberg = f(np.linspace(0.0, 100.0, snaps.shape[0]))
    
    
#    # Construct test vector in submatrix space 
#    indvect=ind[ind_col[pvect]]
#    snap = snaps[:, [pvect]]
#    snapk = [snap[indvect==i] for i in range(nclust[1])]
#    # Project Vk submatrces onto given test vector in submatrix space
#    Vksub = Vk[ind_col[pvect]*nclust[1]:(ind_col[pvect]+1)*nclust[1]]
#    proj = [project(Vksub[i], snapk[i]) for i in range(nclust[1])]
#    # Reconsruct vector in original space
#    projl = Vkvectl=[list(a) for a in proj]
#    projvect = [0]*snaps.shape[0]
#    for i in range(snaps.shape[0]):
#        projvect[i]=projl[indvect[i]].pop(0)
#    
#    # Berg Plotting
#    # plot 1st 5 modes of 1st basis
#    plot_it(Vk[0][:, :5])
#    # plot centers
#    plot_it(ck_col.T)
#    # plot vector and its projection onto nearest basis
#    plot_it(np.hstack((snaps[:, [pvect]], projvect)))
    # plot original vector
    plot_it(snaps[:-1, [pvect]],np.linspace(0.0, 100.0, 1000)[:-1, None])
#    # plot downsampled vectors
    plot_it(downsample,downlocs)
    plot_it(downsample_cberg,downlocs_cberg, [0,100,2,6])
#    print(downsample.shape)
    #plot_it(downsample2,downlocs2)
    # plot projections of downsampled vectors
    #plot_it(proj,downlocs)
    #plot_it(proj_cberg,downlocs_cberg)
#    # plot vector and its projection onto nearest col-only basis
    plot_it(np.hstack((snaps[:, [pvect]], project(Vk_col[ind_col[pvect]], snaps[:, [pvect]]))))
    # plot both downsampled and un-downsampled vectors
    plot_it(np.hstack((snaps[:, [pvect]], projinterp[:,None])))
    plot_it(np.hstack((snaps[:, [pvect]], projinterp_cberg[:,None])))
#    # plot vector and its projection onto 1st col-only basis
##    plot_it(np.hstack((snaps[:, [pvect]], project(Vk_col[0], snaps[:, [pvect]]))))    
    
    # Burg Error Calc
    # Calculate RMS error between vector and projection
    projcolvect = project(Vk_col[ind_col[pvect]], snaps[:, [pvect]])
    Mean_Sqrt_Error_colvect = np.sqrt(np.mean(np.square(projcolvect-snaps[:, [pvect]])))
    Mean_Sqrt_Error_interp = np.sqrt(np.mean(np.square(projinterp[:,None]-snaps[:, [pvect]])))
    Mean_Sqrt_Error_interp_mod = np.sqrt(np.mean(np.square(projinterp_cberg[:,None]-snaps[:, [pvect]])))
    print("--- Mean Abs Error is %s---" %round(Mean_Sqrt_Error_colvect,5))
    print("--- Mean HROM Orig Sqrt Error is %s---" %round(Mean_Sqrt_Error_interp,5))
    print("--- Mean HROM Mod Sqrt Error is %s---" %round(Mean_Sqrt_Error_interp_mod,5))
#
#    # CFD Plotting
#    # plot centers
##    plot_it(ck_col.T[:,0])
#    # plot vector 
##    plot_it(np.hstack((snaps[:, [pvect]])),'uabs')
#    # plot vector and its projection onto nearest basis
##    plot_it(np.hstack((projvect)),'uabs')
#    # plot vector projection onto nearest col-only basis
##    plot_it(np.hstack((project(Vk_col[ind_col[pvect]], snaps[:, [pvect]]))),'uabs')
#    # plot vector projection onto 1st col-only basis
##    plot_it(np.hstack((project(Vk_col[0], snaps[:, [pvect]]))))
#    # plot vector and its projection onto nearest basis
##    plot_ind(np.hstack((indvect)),'ru',nclust[1])
#    
#    # Old CFD Plotting
#    #Choose vector to plot
##    pvect=20
#    # plot centers
##    plot_it(ck[:,0])
#    # plot vector 
##    plot_it(np.hstack((snaps[:, [pvect]])))
#    # plot vector projection onto nearest basis
##    plot_it(np.hstack((project(Vk[ind[pvect]], snaps[:, [pvect]]))))
#    # plot vector projection onto 1st basis
##    plot_it(np.hstack((project(Vk[0], snaps[:, [pvect]]))))
#    
#    #Discarded:
#    #    Vkvectl=[list(a) for a in Vk[ind_col[pvect]*nclust[1]:(ind_col[pvect]+1)*nclust[1]]]
##    Vkvect = [0]*snaps.shape[0]
##    for i in range(snaps.shape[0]):
##        Vkvect[i]=Vkvectl[indvect[i]].pop(0)
##    Vkvect = np.array(Vkvect)
#    # Split Vk for sub-matrix projections
