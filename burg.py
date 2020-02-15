import numpy as np
import matplotlib.pyplot as plt
import os# Figures out the absolute path for you in case your working directory moves around.    

def locations(v):
    loc = np.linspace(0.0, 100.0, v.shape[0])
    return loc

def plot_it(v, v_indices = np.array([0]), pltaxis = None):
    fig = plt.figure(figsize=(6,6)) #12,6
    #ax = fig.add_subplot(111)
    #plt.axis('off')
    if np.sum(v_indices) == 0:
         vspace = np.linspace(0.0, 100.0, 1000)[:, None]
         plt.plot(vspace, v, lw=2)
    else:
         vspace = v_indices
         #plt.plot(vspace, v, lw=2)
         plt.plot(vspace, v, 'ro')
    if pltaxis != None:
         plt.axis(pltaxis)
    #my_path = 'C:\\Users\\Tina\\Desktop\\FRG\\'
    #plt.savefig(my_path + 'test.png')   
    plt.show()
    
def plot_20_its(v):
    fig, axs = plt.subplots(1,20, figsize=(24, 1), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.2)
    axs = axs.ravel()
    for i in range(20):
        axs[i].set_ylim([0,12])
        axs[i].plot(np.linspace(0.0, 100.0, v.shape[0])[:, None], v[:,i], color='w', linewidth=2.5)
        axs[i].set_xticklabels([])
        axs[i].xaxis.set_ticks_position('none') 
        axs[i].set_yticklabels([])
        axs[i].yaxis.set_ticks_position('none') 
        axs[i].patch.set_facecolor('black')
    my_path = 'C:\\Users\\Tina\\Desktop\\FRG\\'
    plt.savefig(my_path + 'test.png')   
    plt.show()
