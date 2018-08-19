from args import DIMS, DIM, DIR
from ripser import ripser
from mnist import MNIST
from src.util import *
#import persim, umap
import numpy as np
import warnings

def fdict(pc):
    b,d,c =  pc[0][0], pc[0][1], pc[1][:,:-1]
    return {'birth' : b, 'death' : d, 'cycles' : c, 't' : d - b}

def azip(r,d):
    return zip(r['dgms'][d], r['cocycles'][d])

def fmap(r, d):
    return sorted(map(fdict, azip(r, d)), key=lambda x: x['t'], reverse=True)

def persist(data, X, c):
    return ripser(data['X'][X[c]].T, do_cocycles=True)

def get_persist(data, dims=DIMS):
    dims = range(dims+1)
    C = np.unique(data['y'])
    X = {c : np.where(data['y'] == c)[0] for c in C}
    sprint(2, '| computing persistence diagrams')
    warnings.filterwarnings("ignore")
    R = dmap(persist, C, data, X)
    # R = {c : ripser(data['X'][X[c]].T, do_cocycles=True) for c in C}
    sprint(2, '| retrieving cocycles')
    B = {d : {c : R[c]['dgms'][d] for c in C} for d in dims}
    D = {d : {c : fmap(R[c], d) for c in C} for d in dims}
    return {'keys' : C, 'data' : data, 'diagrams' : D, 'barcodes' : B} #, 'cocycles' : G}

def to_mask(dgm, k=100, l=0.0, n=28*28):
    c = dgm['cycles'] if len(dgm['cycles']) < k else dgm['cycles'][:k]
    return [dgm['t'] if i in dgm['cycles'][:k] else l for i in range(n)]

# def make_masks(dgm, k=100, l=0.0):
#     return np.vstack(map(lambda x: to_mask(x, k, l), dgm)).reshape(-1, 28, 28)
#
# def masks_fun(dgms, k, l, c):
#     return make_masks(dgms[c], k ,l)

def masks_fun(dgms, k, l, c):
    return np.vstack(map(lambda x: to_mask(x, k, l), dgms[c])).reshape(-1, 28, 28)

def get_masks(jdict, dim=DIM, k=100, l=0.0):
    return dmap(masks_fun, jdict['keys'], jdict['diagrams'][dim], k ,l)
    # x, args = jdict['keys'], (jdict['diagrams'][dim], k ,l)
    # return dict(zip(jdict['keys'], pmap(masks_fun, x, *args)))
    # return dict(zip(jdict['keys'], pmap(f, jdict['keys'])))
    # return {c : make_masks(jdict['diagrams'][dim][c], k ,l, f) for c in jdict['keys']}

def fmask(X):
    x = np.array(np.sum(X, axis=0), dtype=float)
    num, den = x - x.min(), x.max() - x.min()
    return x if x.max() == 0 else x / x.max() if den == 0 else num / den
