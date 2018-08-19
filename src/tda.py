from args import DIMS, DIM, DIR
from ripser import ripser
from mnist import MNIST
from multiprocessing import Pool
from functools import partial
from src.util import sprint
#import persim, umap
import numpy as np
import warnings

LOAD = {'train': lambda x: x.load_training(), 'test': lambda x: x.load_testing()}

def get_data(key='train', dir=DIR):
    return dict(zip(('X','y'), LOAD[key](MNIST(dir, return_type='numpy'))))

def fdict(pc):
    b,d,c =  pc[0][0], pc[0][1], pc[1][:,:-1]
    return {'birth' : b, 'death' : d, 'cycles' : c, 't' : d - b}

def azip(r,d):
    return zip(r['dgms'][d], r['cocycles'][d])

def fmap(r, d):
    return sorted(map(fdict, azip(r, d)), key=lambda x: x['t'], reverse=True)

def persist(data, X, c):
    return ripser(data['X'][X[c]].T, do_cocycles=True)

def pmap(f, x):
    pool = Pool()
    y = pool.map(f, x)
    pool.close()
    pool.join()
    return y


def get_masks(data, dims=DIMS):
    dims = range(dims+1)
    C = np.unique(data['y'])
    X = {c : np.where(data['y'] == c)[0] for c in C}
    sprint(2, '| computing persistence diagrams')
    warnings.filterwarnings("ignore")
    f = partial(persist, data, X)
    R = dict(zip(C, pmap(f, C)))
    # pool = Pool()
    # R = dict(zip(C, pool.map(f, C)))
    # pool.close()
    # pool.join()
    # R = {c : ripser(data['X'][X[c]].T, do_cocycles=True) for c in C}
    sprint(2, '| retrieving cocycles')
    B = {d : {c : R[c]['dgms'][d] for c in C} for d in dims}
    D = {d : {c : fmap(R[c], d) for c in C} for d in dims}
    return {'keys' : C, 'data' : data, 'diagrams' : D, 'barcodes' : B} #, 'cocycles' : G}

def to_mask(dgm, k=100, l=0.0, f=lambda x: x ** 2, n=28*28):
    c = dgm['cycles'] if len(dgm['cycles']) < k else dgm['cycles'][:k]
    return [f(dgm['t']) if i in dgm['cycles'][:k] else l for i in range(n)]

def make_masks(dgm, k=100, l=0.0, f=lambda x: x ** 2):
    return np.vstack(map(lambda x: to_mask(x, k, l, f), dgm)).reshape(-1, 28, 28)

def masks_fun(dgms, k, l, f, c):
    return make_masks(dgms[c], k ,l, f)

def fmask(X):
    x = np.array(np.sum(X, axis=0), dtype=float)
    num, den = x - x.min(), x.max() - x.min()
    return x if x.max() == 0 else x / x.max() if den == 0 else num / den
    # mx, mn = x.max(), x.min()
    # if mx == 0:
    # elif mx - mn == 0:
    #     return x / mx
    # return (x - mx) / (mx - mn)
    # return x if mx == 0 else x / mx if mx - mn == 0 else (x - mx) / (mx - mn)

def build_masks(jdict, dim=DIM, k=100, l=0.0, f=lambda x: x ** 2):
    f = partial(masks_fun, jdict['diagrams'][dim], k ,l, f)
    return dict(zip(jdict['keys'], pmap(f, jdict['keys'])))
    # pool = Pool()
    # d = dict(zip(jdict['keys'], pool.map(f, jdict['keys'])))
    # pool.close()
    # pool.join()
    # return d
    # return {c : make_masks(jdict['diagrams'][dim][c], k ,l, f) for c in jdict['keys']}
