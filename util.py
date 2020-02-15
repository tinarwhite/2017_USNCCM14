import numpy as np

def read_me(*args):
    return np.hstack([np.loadtxt(x) for x in args])
