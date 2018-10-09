from args import DIMS, DIM, DIR
from ripser import ripser
from mnist import MNIST
from src.util import *
import numpy as np
import warnings

def fdict(pc):
    b,d,c =  pc[0][0], pc[0][1], pc[1][:,:-1]
    return {'birth' : b, 'death' : d, 'cycles' : c, 't' : d - b}

def azip(r,d):
    return zip(r['dgms'][d], r['cocycles'][d])

def fmap(r, d):
    return sorted(map(fdict, azip(r, d)), key=lambda x: x['t'], reverse=True)

def persist(data, X, dim, c):
    return ripser(data['X'][X[c]].T, do_cocycles=True, maxdim=dim)

def get_persist(data, dim=DIMS, n=10):
    dims = range(dim+1)
    C = np.unique(data['y'])
    X = {c : np.where(data['y'] == c)[0] for c in C}
    sprint(2, '| computing persistence diagrams')
    warnings.filterwarnings("ignore")
    R = dmap(persist, C, data, X, dim)
    sprint(2, '| retrieving cocycles')
    B = {d : {c : R[c]['dgms'][d] for c in C} for d in dims}
    D = {d : {c : fmap(R[c], d) for c in C} for d in dims}
    return {'keys' : C, 'data' : data, 'diagrams' : D, 'barcodes' : B}

def to_mask(dgm, k, l, n):
    c = dgm['cycles'] if len(dgm['cycles']) < k else dgm['cycles'][:k]
    return [dgm['t'] if i in dgm['cycles'][:k] else l for i in range(n)]

def masks_fun(dgms, k, l, s, c):
    return np.vstack(map(lambda x: to_mask(x, k, l, s[0]*s[1]), dgms[c])).reshape(-1, *s)

def get_masks(jdict, dim=DIM, k=100, l=0.0):
    return dmap(masks_fun, jdict['keys'], jdict['diagrams'][dim], k ,l, jdict['data']['shape'])

def fmask(X):
    y = np.array(np.sum(X, axis=0), dtype=float)
    # for i in range(y.shape[2]):
    #     x = y[:,:,i]
    #     num, den = x - x.min(), x.max() - x.min()
    #     y[:,:,i] = x if x.max() == 0 else x / x.max() if den == 0 else num / den
    return y

def build(args, dat, i):
    sprint(0, '[ constructing masks for channel %d' % i)
    sprint(1, '[ getting masks in dimension 0-%d' % args.dims)
    jdict = get_persist(dat, args.dims)
    sprint(1, '[ building masks in dimension %d' % args.dim)
    jdict['masks'] = get_masks(jdict, args.dim, args.k)
    return jdict

def build_mask(jdicts, c):
    masks = [jdict['masks'][c] for jdict in jdicts]
    n = min(m.shape[0] for m in masks)
    return np.stack([m[:n] for m in masks], 3)
