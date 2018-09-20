from multiprocessing import Pool
from functools import partial
import numpy as np
import os, sys, gc

def sprint(i, s):
    print(i*' ' + s)

def pmap(fun, x, *args):
    pool = Pool()
    f = partial(fun, *args)
    try:
        y = pool.map(f, x)
    except KeyboardInterrupt as e:
        print(e)
        pool.close()
        pool.join()
        sys.exit()
    pool.close()
    pool.join()
    return y

def dmap(fun, x, *args):
    return dict(zip(x, pmap(fun, x, *args)))

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# def get_stats(data): # , axis=0):
#     mean = np.mean(data, axis=0)
#     std = np.std(data, axis=0, ddof=0)
#     return mean, std

def get_stats(data): # , axis=0):
    mean = np.mean(data)
    std = np.std(data, ddof=0)
    return mean, std

# def standardize(data, stats): # , axis=0):
#     mean, std = stats
#     for i,s in enumerate(stats):
#         x = data[:,i]
#         data[:,i] = (x - mean[i]) / (1. if std[i] == 0 else std[i])
#         # x = np.take(data, i, axis)
#         # y = (x - mean[i]) / (1. if std[i] == 0 else std[i])
#         # np.put_along_axis(data, i, y, axis)
#     return data

# def all_stats(data, masks):
#     # fmasks = map(fmask, [np.swapaxes(np.swapaxes(masks[c], 1, 3), 2, 3) for c in range(10)])
#     # fmasks = map(fmask, masks)
#     X = np.array([x['X'] for x in data])
#     M = np.array(masks).reshape(len(masks)*len(data), X.shape[2])
#     data = None
#     gc.collect()
#     sprint(1, '[ computing masked tensor')
#     sprint(2, '| data\t'+str(X.shape))
#     sprint(2, '| mask\t'+str(M.shape))
#     MX = np.array([[mask * x for x in X] for mask in M])
#     sprint(1, '[ computing stats for '+str(MX.shape)+' tensor')
#     stats = [get_stats(x) for x in MX]
#     # return dict(zip(('mean','std'), zip(*stats)))
#     return zip(*stats)

def mask_stats(x, m):
    x = x * m
    return np.mean(x), np.std(x, ddof=0)

def all_stats(data, masks):
    # fmasks = map(fmask, [np.swapaxes(np.swapaxes(masks[c], 1, 3), 2, 3) for c in range(10)])
    # fmasks = map(fmask, masks)
    n, m = len(masks), len(data)
    X = np.array([x['X'] for x in data])
    M = np.array(masks).reshape(n, m, -1)
    # M = np.array(masks).reshape(len(masks)*len(data), X.shape[2])
    # M = np.array(masks).reshape(len(data), len(masks), X.shape[2])
    data = None
    gc.collect()
    sprint(1, '[ computing masked tensor')
    sprint(2, '| data\t'+str(X.shape))
    sprint(2, '| mask\t'+str(M.shape))
    return np.array([[mask_stats(X[i], M[j,i]) for i in range(m)] for j in range(n)])
    # np.array([x * m for m in mask])
    # means = [np.mean(np.array)]
    # sprint(1, '[ computing stats for '+str(MX.shape)+' tensor')
    # stats = [get_stats(x) for x in MX]
    # return dict(zip(('mean','std'), zip(*stats)))
    # return zip(*stats)
