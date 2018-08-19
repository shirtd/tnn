from multiprocessing import Pool
from functools import partial
import os, sys

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
