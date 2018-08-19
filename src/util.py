from multiprocessing import Pool
from functools import partial

def sprint(i, s):
    print(i*' ' + s)

def pmap(fun, x, *args):
    pool = Pool()
    f = partial(fun, *args)
    y = pool.map(f, x)
    pool.close()
    pool.join()
    return y

def dmap(fun, x, *args):
    return dict(zip(x, pmap(fun, x, *args)))
