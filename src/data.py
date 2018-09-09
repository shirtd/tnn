from datasets import cifar, mnist
from args import DIR, SRC
from mnist import MNIST
import pandas as pd
import numpy as np
import os

LOAD = {'train': lambda x: x.load_training(), 'test': lambda x: x.load_testing()}

def get_mnist(key='train', dir=DIR, src=SRC):
    dir = os.path.join(src, dir)
    X,y = LOAD[key](MNIST(dir, return_type='numpy'))
    return [{'X' : X, 'y' : y, 'shape' : (28, 28)}], mnist.CLASS

def get_cifar(key='train', dir=DIR, src=SRC):
    dir = 'cifar-10'
    fname = '_'.join(['cifar', key]) + '.csv'
    fpath = os.path.join(src, dir, fname)
    df = pd.read_csv(fpath)
    X = df.drop(cifar.ICOLS, axis=1).values
    y = df.iloc[:,cifar.GT].values
    return [{'X' : X[:,(i*1024):((i+1)*1024)], 'y' : y, 'shape' : (32, 32)} for i in range(3)], cifar.CLASS

DATASETS = {'mnist' : get_mnist, 'cifar' : get_cifar}

def get_data(name='mnist', key='train', dir=DIR, src=SRC):
    return DATASETS[name](key, dir)
