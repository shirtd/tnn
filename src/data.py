from torchvision.datasets import MNIST, CIFAR10
from torchvision import datasets, transforms
from datasets import cifar, mnist
from args import DIR, SRC
import pandas as pd
import numpy as np
import os

def load_data(x, key='train'):
    D = x('../data', train=True, transform=transforms.ToTensor())
    if key == 'train':
        return np.array(D.train_data), np.array(D.train_labels)
    else:
        return np.array(D.test_data), np.array(D.test_labels)

LOAD = {'train': lambda x: load_data(x, 'train'),
        'test': lambda x: load_data(x, 'test')}

def get_mnist(key='train'):
    X, y = LOAD[key](MNIST)
    n, i, j = X.shape
    return [{'X' : X.reshape(n, i*j), 'y' : y, 'shape' : (i, j)}], mnist.CLASS

def get_cifar(key='train'):
    X, y = LOAD[key](CIFAR10)
    n,i,j,k = X.shape
    return [{'X' : X[:,:,:,l].reshape(n, i*j), 'y' : y, 'shape' : (i, j)} for l in range(k)], cifar.CLASS

DATASETS = {'mnist' : get_mnist, 'cifar' : get_cifar}

def get_data(name='mnist', key='train'):
    return DATASETS[name](key)
