from args import DIMS, DIM, DIR
from ripser import ripser
from mnist import MNIST
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

def get_masks(data, dims=DIMS):
    dims = range(dims+1)
    C = np.unique(data['y'])
    X = {c : np.where(data['y'] == c)[0] for c in C}
    print(' | computing persistence diagrams')
    warnings.filterwarnings("ignore")
    R = {c : ripser(data['X'][X[c]].T, do_cocycles=True) for c in C}
    print(' | retrieving cocycles')
    B = {d : {c : R[c]['dgms'][d] for c in C} for d in dims}
    D = {d : {c : fmap(R[c], d) for c in C} for d in dims}
    return {'keys' : C, 'data' : data, 'diagrams' : D, 'barcodes' : B} #, 'cocycles' : G}

def to_mask(dgm, l=0.0, n=28*28):
    return [dgm['t'] if i in dgm['cycles'] else l for i in range(n)]

def make_masks(dgm):
    return np.vstack(map(to_mask, dgm)).reshape(-1,28,28)

def fmask(x):
    return np.sum(x, axis=0)

def build_masks(jdict, dim=DIM):
    return {c : make_masks(jdict['diagrams'][dim][c]) for c in jdict['keys']}
