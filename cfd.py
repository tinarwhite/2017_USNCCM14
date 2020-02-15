import numpy as np
import matplotlib.pyplot as plt

from util import read_me

ref = 0
if ref == 0:
    ns, nv, nc, nt = 3, 513, 3, 978
elif ref == 1:
    ns, nv, nc, nt = 3, 2004, 3, 3912

def post(v, which):
    v = v.reshape((ns, nc, nt), order='F')
    if which == 'r':
        Q = v[:, 0, :]
    if which == 'ru':
        Q = v[:, 1, :]
    if which == 'rv':
        Q = v[:, 2, :]
    if which == 'uabs':
        Q = np.sqrt((v[:, 1, :]/v[:, 0, :])**2+(v[:, 2, :]/v[:, 0, :])**2) 
    return Q.ravel('F')

def plot_it(v, which='uabs'):
    fig = plt.figure()
    ax = fig.add_subplot(111).set_aspect('equal')
    p = np.loadtxt('cfd/naca0012ref{0:d}p1.nodes'.format(ref))
    #t       np.loadtxt('cfd/naca0012ref{0:d}p1.elem'.format(ref))
    Q = post(v, which)
    p = p.reshape((nt, 2, ns))
    t = np.arange(0, 3*nt).reshape(nt, 3, order='C')
    #print p.shape, Q.shape
    #t = np.hstack((np.arange(0, nt-2), np.arange(1, nt-1), np.arange(2, nt)))
    #Q = p[:, 0, :].reshape(ns*nt, order='C')
    plt.tricontourf(p[:, 0, :].reshape(ns*nt, order='C'),
                    p[:, 1, :].reshape(ns*nt, order='C'), t, Q)
    #plt.tricontourf(p[:, 0], p[:, 1], t, Q)
    plt.show()

if __name__ == '__main__':
    v = read_me('cfd/naca0012ref{0:d}p1.snaps.mu2'.format(ref))
    plot_it(v[:, 50], 'uabs')
    #print v.shape
    #p = np.loadtxt('cfd/naca0012ref1p1.nodes')
    #print p.reshape((nt, 2, ns)).shape
    #print np.arange(0, 3*nt).reshape((nt, 3), order='C')
