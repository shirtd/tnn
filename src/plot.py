import matplotlib.pyplot as plt
from persim import PersImage
from util import *
from tda import *
import os

def plot_mask(axis, x, f=fmask):
    s = f(x)
    if s.shape[2] != 3:
        s = s[:, :, 0]
    axis.imshow(s)
    return s

def plot_masks(axis, masks):
    return map(lambda (c,m): plot_mask(axis[c/5,c%5], m), masks.iteritems())
    # return [plot_mask(ax[c/5,c%5], m) for c,m in masks.iteritems()]

with HiddenPrints():
    pim = PersImage()

def plot_img(axis, x, c=None, clear=False):
    img = pim.transform(x)
    axis.imshow(img, interpolation="bilinear")
    return img

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
    # map(lambda (c,d): plot_dgm(axis[c/5,c%5], d), dgms[1].iteritems())

def plot_imgs(axis, dgms):
    map(lambda (c,d): plot_img(axis[c/5,c%5], d), dgms[1].iteritems())

plot_persist = {'img' : plot_imgs, 'dgm' : plot_dgms}

plt.ion()
def axinit(i, r=2, c=5, sharex=True, sharey=True, figsize=(12,5)):
    fig, ax = plt.subplots(r, c, sharex, sharey, figsize=figsize, num=i)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0., hspace=0.)
    return fig, ax

fig, ax = zip(*map(axinit,range(2)))

def plot(args, jdicts, masks):
    for jdict in jdicts:
        plot_persist[args.plot](ax[0], jdict['barcodes'])
    plot_masks(ax[1], masks)
    if args.save:
        if not os.path.exists(args.pdir):
            sprint(2, '! creating directory %s' % args.pdir)
            os.mkdir(args.pdir)
        fdgm = os.path.join(args.pdir, '_'.join([args.data,'%s.png' % args.plot]))
        fmask = os.path.join(args.pdir, '_'.join([args.data,'masks.png']))
        sprint(2, '| saving %s' % fdgm)
        fig[0].savefig(fdgm)
        sprint(2, '| saving %s' % fmask)
        fig[1].savefig(fmask)
        return fdgm, fmask
