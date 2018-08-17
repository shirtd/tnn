import matplotlib.pyplot as plt
from tda import *

def plot_mask(axis, x, f=fmask):
    s = f(x)
    axis.imshow(s, interpolation="bilinear")
    return s

def plot_masks(axis, masks):
    return map(lambda (c,m): plot_mask(axis[c/5,c%5], m), masks.iteritems())
    # return [plot_mask(ax[c/5,c%5], m) for c,m in masks.iteritems()]

def plot_dgm(axis, x, c=None, clear=False):
    if clear:
        axis.cla()
    if c != None:
        axis.scatter(x[:,0], x[:,1], c=c, s=5, alpha=0.5)
    else:
        axis.scatter(x[:,0], x[:,1], s=5, alpha=0.5)

def plot_dgms(axis, dgms):
    map(lambda (c,d): plot_dgm(axis[c/5,c%5], d, 'blue', True), dgms[0].iteritems())
    map(lambda (c,d): plot_dgm(axis[c/5,c%5], d, 'red'), dgms[1].iteritems())
    # return [plot_mask(ax[c/5,c%5], m) for c,m in masks.iteritems()]

plt.ion()
def axinit(i, r=2, c=5, sharex=True, sharey=True, figsize=(12,5)):
    fig, ax = plt.subplots(r, c, sharex, sharey, figsize=figsize, num=i)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0., hspace=0.)
    return fig, ax

fig, ax = zip(*map(axinit,range(2)))
